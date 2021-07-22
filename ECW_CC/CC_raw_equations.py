'''
Raw CC equations in Spin-orbital basis 
- CCS
- CCSD
- CCSDT

T and Lambda (from PySCF ccn.equation)
R and L
'''

from numpy import einsum as e
from pyscf.ccn.util import p

##########
# CCS
##########

def energy_s(t1, eris):

    nocc,nvir = t1.shape
    ov = eris.fock[:nocc,nocc:].copy()

    scalar = 0
    scalar += e("ia,ia", ov, t1)  # s0
    scalar += 1./2 * e("jiba,ia,jb", eris.oovv, t1, t1)  # s1

    return scalar

def T1eq(t1, eris, fsp=None):
    """
    PySCF T1 equations
    Check against Stasis: same if eris ijab = -jiab

    :param t1:
    :param eris:
    :param fsp:
    :return:
    """

    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    oo = f[:nocc, :nocc].copy()
    vo = f[nocc:, :nocc].copy()
    ov = f[:nocc, nocc:].copy()
    vv = f[nocc:, nocc:].copy()

    hE = 0
    hE += e("ba,ia->ib", vv, t1)  # d0_ov
    hE -= e("ij,ia->ja", oo, t1)  # d1_ov
    hE += e("ai->ia", vo)  # d2_ov
    hE -= e("ia,ib,ja->jb", ov, t1, t1)  # d3_ov
    hE -= e("jacb,jb,ic->ia", eris.ovvv, t1, t1)  # d4_ov
    hE += e("jabi,jb->ia", eris.ovvo, t1)  # d5_ov
    hE -= e("jkbc,jb,ka,ic->ia", eris.oovv, t1, t1, t1)  # d6_ov
    hE -= e("jiak,ja,ib->kb", eris.oovo, t1, t1)  # d7_ov

    return hE

def La1eq(t1, l1, eris,fsp=None):
    """
    Lambda 1 equation from PySCF ccn module

    :param t1:
    :param l1:
    :param eris:
    :param fsp:
    :return:
    """
    
    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock
    else:
        f = fsp

    vv = f[nocc:, nocc:].copy()
    oo = f[:nocc, :nocc].copy()
    ov = f[:nocc, nocc:].copy()

    He = 0
    He += e("ba,ib->ia", vv, l1)  # d0_ov
    He -= e("ji,ja->ia", oo, l1)  # d1_ov
    He += e("ia->ia", ov)  # d2_ov
    He -= e("jb,ia,ja->ib", ov, l1, t1)  # d3_ov
    He -= e("jb,ia,ib->ja", ov, l1, t1)  # d4_ov
    He -= e("jabc,ia,ib->jc", eris.ovvv, l1, t1)  # d5_ov
    He += e("jabc,ia,jb->ic", eris.ovvv, l1, t1)  # d6_ov
    He += e("jabi,ia->jb", eris.ovvo, l1)  # d7_ov
    He += e("ijab,ia->jb", eris.oovv, t1)  # d8_ov
    He += e("kiab,jc,ia,kc->jb", eris.oovv, l1, t1, t1)  # d9_ov
    He += e("ikba,jc,ia,jb->kc", eris.oovv, l1, t1, t1)  # d10_ov
    He -= e("jkac,ib,ia,jb->kc", eris.oovv, l1, t1, t1)  # d11_ov
    He -= e("jkbi,ia,jb->ka", eris.oovo, l1, t1)  # d12_ov
    He -= e("kibj,ja,ia->kb", eris.oovo, l1, t1)  # d13_ov

    return He

def R1eq(t1, r1, r0, eris, fsp=None):
    # commented lines are terms to be included if f are kinetic operator elements

    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    vv = f[nocc:, nocc:].copy()
    oo = f[:nocc, :nocc].copy()
    vo = f[nocc:, :nocc].copy()
    ov = f[:nocc, nocc:].copy()

    # no t terms
    Ria  = r0*e('ai->ia', vo)
    Ria += e('ab,ib->ia', vv, r1)
    Ria -= e('ji,ja->ia', oo, r1)
    Ria += e('jb,ajib->ia', r1, eris.voov)
    # Ria -= e('ja,jkik->ia', r1, eris.oooo)  ##
    # Ria += e('ib,akbk->ia', r1, eris.vovo)  ##
    # Ria += r0*e('akik->ia', eris.vooo)  ##

    # t terms
    Ria += e('jb,jb,ia->ia', t1, ov, r1)
    Ria -= e('ib,jb,ja->ia', t1, ov, r1)
    Ria -= e('ja,jb,ib->ia', t1, ov, r1)
    Ria += r0*e('ib,ab->ia', t1, vv)
    Ria -= r0*e('ja,ji->ia', t1, oo)
    Ria += r0*e('jb,jabi->ia', t1, eris.ovvo)
    # Ria -= r0*e('ja,jkik->ia', t1, eris.oooo)  ##
    # Ria += r0*e('ib,akbk->ia', t1, eris.vovo)  ##
    # Ria += e('ia,jb,jkbk->ia', r1, t1, eris.oovo)
    Ria -= e('ka,jb,jkbi->ia', r1, t1, eris.oovo)
    Ria += e('ib,jc,jacb->ia', r1, t1, eris.ovvv)
    # Ria -= e('ib,ja,jkbk->ia', r1, t1, eris.oovo)  ##
    Ria -= e('kc,ja,jkic->ia', r1, t1, eris.ooov)
    # Ria -= e('ka,ib,kjbj->ia', r1, t1, eris.oovo)  ##
    Ria += e('kc,ib,akbc->ia', r1, t1, eris.vovv)

    # t.t terms
    Ria -= r0 * e('ib,ja,jb->ia', t1, t1, ov)
    Ria -= r0*e('jb,ka,jkbi->ia', t1, t1, eris.oovo)
    Ria += r0*e('jb,ic,jabc->ia', t1, t1, eris.ovvv)
    # Ria -= r0*e('ib,ka,kjbj->ia', t1, t1, eris.oovo)  ##
    Ria += 0.5*e('ia,jb,kc,jkbc->ia', r1, t1, t1, eris.oovv)
    Ria -= e('ka,jb,ic,jkbc->ia', r1, t1, t1, eris.oovv)
    Ria -= e('ic,jb,ka,jkbc->ia', r1, t1, t1, eris.oovv)
    Ria -= e('jc,ib,ka,kjbc->ia', r1, t1, t1, eris.oovv)
    
    # t.t.t
    Ria += r0*e('ja,kb,ic,jkbc->ia', t1, t1, t1, eris.oovv)

    return Ria

def R10eq(t1, r1, r0, eris, fsp=None):
    
    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    ov = f[:nocc,nocc:].copy()

    # no t
    R0  = e("ia,ia", ov, r1)
    #R0 += e("jb,jkbk", r1, eris.oovo)

    # t
    R0 += r0*e("ia,ia", t1, ov)
    #R0 += r0*e("jb,jkbk", t1, eris.oovo)
    R0 += e("kc,jb,jkbc", r1, t1, eris.oovv)

    # t.t
    R0 += 0.5*r0*e("jb,kc,jkbc", t1, t1, eris.oovv)

    return R0

def es_L1eq(t1, l1, l0, eris, fsp=None):

    nocc, nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    vv = f[nocc:, nocc:].copy()
    oo = f[:nocc, :nocc].copy()
    ov = f[:nocc, nocc:].copy()

    # no t terms
    Lia = l0*ov
    Lia += e('ba,ib->ia', vv, l1)
    Lia -= e('ij,ja->ia', oo, l1)
    Lia += e('jb,bija->ia', l1, eris.voov)
    #Lia -= e('ja,ikjk->ia', l1, eris.oooo)  ##
    #Lia += e('ib,bkak->ia', l1, eris.vovo)  ##
    #Lia += l0*e('ikak->ia', eris.oovo)  ##

    # t terms
    Lia += e('jb,jb,ia->ia', t1, ov, l1)
    Lia -= e('jb,ib,ja->ia', t1, ov, l1)
    Lia -= e('jb,ja,ib->ia', t1, ov, l1)
    Lia += l0*e('jb,jiba->ia', t1, eris.oovv)
    #Lia += e('ia,jb,jkbk->ia', l1, t1, eris.oovo)  ##
    Lia -= e('ka,jb,jibk->ia', l1, t1, eris.oovo)
    Lia += e('ic,jb,jcba->ia', l1, t1, eris.ovvv)
    #Lia -= e('ib,jb,jkak->ia', l1, t1, eris.oovo)  ##
    Lia -= e('kc,jc,jika->ia', l1, t1, eris.ooov)
    #Lia -= e('ka,kb,ijbj->ia', l1, t1, eris.oovo)
    Lia += e('kc,kb,ciba->ia', l1, t1, eris.vovv)

    # t.t terms
    Lia += 0.5 * e('ia,jb,kc,jkbc->ia', l1, t1, t1, eris.oovv)
    Lia -= e('ja,kb,jc,kibc->ia', l1, t1, t1, eris.oovv)
    Lia -= e('ic,jb,kc,jkba->ia', l1, t1, t1, eris.oovv)
    Lia -= e('jc,jb,kc,kiba->ia', l1, t1, t1, eris.oovv)

    return Lia

def es_L10eq(t1, l1, l0, eris, fsp=None):
    
    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    ov = f[:nocc, nocc:].copy()
    vo = f[nocc:, :nocc].copy()
    vv = f[nocc:, nocc:].copy()
    oo = f[:nocc, :nocc].copy()

    # no t
    L0  = e('ai,ia', vo, l1)
    #L0 += e('ia,akik', l1, eris.vooo)

    # t
    L0 += e('ia,ib,ab', l1, t1, vv)
    L0 += l0*e('ia,ia', t1, ov)
    L0 -= e('ia,ja,ji', l1, t1, oo)
    #L0 += l0*e('jb,jkbk', t1, eris.oovo)
    L0 += e('ia,jb,jabi', l1, t1, eris.ovvo)
    #L0 -= e('ia,ja,jkik', l1, t1, eris.oooo)
    #L0 += e('ia,ib,akbk',l1, t1, eris.vovo)

    # t.t
    L0 -= e('ib,ia,jb,ja', l1, t1, t1, ov)
    L0 += 0.5*l0*e('jb,kc,jkbc', t1, t1, eris.oovv)
    L0 -= e('ia,jb,ka,jkbi', l1, t1, t1, eris.oovo)
    L0 += e('ia,jb,ic,jabc', l1, t1, t1, eris.ovvv)
    #L0 -= e('ia,ib,ka,kjbj', l1, t1, t1, eris.oovo)

    # t.t.t
    L0 += e('ia,ka,jb,ic,kjbc', l1, t1, t1, t1, eris.oovv)

    return L0

###############
# CCSD and CCD
###############

def energy_sd(t1, t2, eris, fsp=None):

    nocc,nvir = t1.shape
    if fsp is None:
        ov = eris.fock[:nocc, nocc:]
    else:
        ov = fsp[:nocc, nocc:]

    scalar = 0
    scalar += e("ia,ia", ov, t1)  # s0
    scalar += 1./2 * e("jiba,ia,jb", eris.oovv, t1, t1)  # s1
    scalar += 1./4 * e("ijba,ijba", eris.oovv, t2)  # s2
    return scalar

def energy_d(t2, eris):
    scalar = 0
    scalar += 1./4 * e("ijba,ijba", eris.oovv, t2)  # s0
    return scalar

def T1T2eq(t1, t2, eris, fsp=None, equation=True):
    
    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    vv = f[nocc:,nocc:].copy()
    oo = f[:nocc,:nocc].copy()
    vo = f[nocc:,:nocc].copy()
    ov = f[:nocc,nocc:].copy()

    hE = hhEE = 0
    hE += e("ba,ia->ib", vv, t1)  # d0_ov
    hhEE += p("..ab", e("ba,ijac->ijbc", vv, t2))  # d1_oovv
    hE -= e("ij,ia->ja", oo, t1)  # d2_ov
    hhEE -= p("ab..", e("ik,ijba->kjba", oo, t2))  # d3_oovv
    hE += e("ai->ia", vo)  # d4_ov
    hE -= e("jb,ja,ib->ia", ov, t1, t1)  # d5_ov
    hhEE -= p("ab..", e("ic,kc,ijba->kjba", ov, t1, t2))  # d6_oovv
    hhEE -= p("..ab", e("ia,ib,jkac->jkbc", ov, t1, t2))  # d7_oovv
    hE += e("ia,ijab->jb", ov, t2)  # d8_ov
    hhEE += 1./2 * p("ab..", e("cbad,ja,id->jicb", eris.vvvv, t1, t1))  # d9_oovv
    hhEE += 1./2 * e("dcba,ijba->ijdc", eris.vvvv, t2)  # d10_oovv
    hhEE -= p("ab..", e("baci,jc->ijba", eris.vvvo, t1))  # d11_oovv
    hhEE += e("baij->ijba", eris.vvoo)  # d12_oovv
    hhEE -= 1./2 * p("ab..", p("..ab", e("icdb,ia,jb,kd->kjac", eris.ovvv, t1, t1, t1)))  # d13_oovv
    hE += e("jbca,ia,jc->ib", eris.ovvv, t1, t1)  # d14_ov
    hhEE += p("ab..", p("..ab", e("kacd,jd,kicb->ijba", eris.ovvv, t1, t2)))  # d15_oovv
    hhEE += 1./2 * p("..ab", e("kcba,kd,ijba->ijcd", eris.ovvv, t1, t2))  # d16_oovv
    hhEE -= p("..ab", e("kadc,kc,ijbd->ijba", eris.ovvv, t1, t2))  # d17_oovv
    hE += 1./2 * e("icba,ijba->jc", eris.ovvv, t2)  # d18_ov
    hhEE += p("ab..", p("..ab", e("kcaj,ia,kb->ijcb", eris.ovvo, t1, t1)))  # d19_oovv
    hE += e("jabi,jb->ia", eris.ovvo, t1)  # d20_ov
    hhEE += p("ab..", p("..ab", e("ibaj,ikac->kjcb", eris.ovvo, t2)))  # d21_oovv
    hhEE += p("..ab", e("kbij,ka->ijba", eris.ovoo, t1))  # d22_oovv
    hhEE += 1./4 * p("ab..", p("..ab", e("jlca,ia,jb,kc,ld->kibd", eris.oovv, t1, t1, t1, t1)))  # d23_oovv
    hhEE -= 1./4 * p("ab..", e("jkda,ia,ld,jkcb->ilcb", eris.oovv, t1, t1, t2))  # d24_oovv
    hE += e("jkca,ia,jb,kc->ib", eris.oovv, t1, t1, t1)  # d25_ov
    hhEE -= p("ab..", p("..ab", e("kldc,jc,la,kidb->ijba", eris.oovv, t1, t1, t2)))  # d26_oovv
    hhEE += p("ab..", e("licd,lc,kd,ijba->jkba", eris.oovv, t1, t1, t2))  # d27_oovv
    hE -= 1./2 * e("jkbc,ic,jkba->ia", eris.oovv, t1, t2)  # d28_ov
    hhEE -= 1./4 * p("..ab", e("klba,kc,ld,ijba->ijdc", eris.oovv, t1, t1, t2))  # d29_oovv
    hhEE += p("..ab", e("lkcd,ka,lc,ijdb->ijba", eris.oovv, t1, t1, t2))  # d30_oovv
    hE -= 1./2 * e("kjcb,ja,kicb->ia", eris.oovv, t1, t2)  # d31_ov
    hE += e("jiba,jb,ikac->kc", eris.oovv, t1, t2)  # d32_ov
    hhEE -= 1./2 * p("..ab", e("ijab,klbc,ijad->kldc", eris.oovv, t2, t2))  # d33_oovv
    hhEE += 1./4 * e("ijba,klba,ijdc->kldc", eris.oovv, t2, t2)  # d34_oovv
    hhEE += 1./2 * p("ab..", e("jiba,ikba,jldc->kldc", eris.oovv, t2, t2))  # d35_oovv
    hhEE += 1./2 * p("ab..", p("..ab", e("ljdb,jiba,lkdc->ikac", eris.oovv, t2, t2)))  # d36_oovv
    hhEE -= 1./2 * p("ab..", p("..ab", e("ijcl,ia,jb,kc->lkab", eris.oovo, t1, t1, t1)))  # d37_oovv
    hhEE -= 1./2 * p("ab..", e("ijak,la,ijcb->klcb", eris.oovo, t1, t2))  # d38_oovv
    hE += e("jkbi,kb,ja->ia", eris.oovo, t1, t1)  # d39_ov
    hhEE -= p("ab..", p("..ab", e("jkai,kc,jlab->ilcb", eris.oovo, t1, t2)))  # d40_oovv
    hhEE += p("ab..", e("ilck,lc,ijba->kjba", eris.oovo, t1, t2))  # d41_oovv
    hE -= 1./2 * e("ijak,ijab->kb", eris.oovo, t2)  # d42_ov
    hhEE -= 1./2 * p("..ab", e("klij,lb,ka->ijba", eris.oooo, t1, t1))  # d43_oovv
    hhEE += 1./2 * e("klij,klba->ijba", eris.oooo, t2)  # d44_oovv
    return hE, hhEE

def T2eq(t2, eris, fsp=None):
    
    nocc = t2.shape[0]
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    vv = f[nocc:,nocc:].copy()
    oo = f[:nocc,:nocc].copy()

    hhEE = 0
    hhEE += p("..ab", e("ab,ijbc->ijac", vv, t2))  # d0_oovv
    hhEE -= p("ab..", e("ki,kjba->ijba", oo, t2))  # d1_oovv
    hhEE += 1./2 * e("dcba,ijba->ijdc", eris.vvvv, t2)  # d2_oovv
    hhEE += e("baij->ijba", eris.vvoo)  # d3_oovv
    hhEE += p("ab..", p("..ab", e("icak,ijab->kjcb", eris.ovvo, t2)))  # d4_oovv
    hhEE -= 1./2 * p("..ab", e("klad,ijab,kldc->ijbc", eris.oovv, t2, t2))  # d5_oovv
    hhEE += 1./4 * e("kldc,ijdc,klba->ijba", eris.oovv, t2, t2)  # d6_oovv
    hhEE -= 1./2 * p("ab..", e("ildc,ijba,lkdc->jkba", eris.oovv, t2, t2))  # d7_oovv
    hhEE += 1./2 * p("ab..", p("..ab", e("jiba,jkbc,ilad->klcd", eris.oovv, t2, t2)))  # d8_oovv
    hhEE += 1./2 * e("ijkl,ijba->klba", eris.oooo, t2)  # d9_oovv
    return hhEE

def La1La2eq(t1, t2, a1, a2, eris, fsp=None):
    
    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f =  fsp

    vv = f[nocc:,nocc:].copy()
    oo = f[:nocc,:nocc].copy()
    vo = f[nocc:,:nocc].copy()
    ov = f[:nocc,nocc:].copy()

    He = HHee = 0
    He += e("ab,ia->ib", vv, a1)  # d0_ov
    HHee -= p("..ab", e("ac,ijab->ijbc", vv, a2))  # d1_oovv
    He += e("ac,ijab,ic->jb", vv, a2, t1)  # d2_ov
    He -= e("ji,ia->ja", oo, a1)  # d3_ov
    HHee -= p("ab..", e("ik,kjba->ijba", oo, a2))  # d4_oovv
    He -= e("jk,kiba,jb->ia", oo, a2, t1)  # d5_ov
    He += e("ai,ijab->jb", vo, a2)  # d6_ov
    He += e("ia->ia", ov)  # d7_ov
    HHee += p("..ab", p("ab..", e("jb,ia->ijab", ov, a1)))  # d8_oovv
    He -= e("ja,ib,jb->ia", ov, a1, t1)  # d9_ov
    He -= e("jb,ia,ib->ja", ov, a1, t1)  # d10_ov
    HHee += p("..ab", e("kc,ijba,kb->ijac", ov, a2, t1))  # d11_oovv
    He -= e("ka,ijcb,ia,kc->jb", ov, a2, t1, t1)  # d12_ov
    HHee -= p("ab..", e("jc,ikba,kc->ijba", ov, a2, t1))  # d13_oovv
    He += e("ia,jkbc,kica->jb", ov, a2, t2)  # d14_ov
    He -= 1./2 * e("jc,ikba,ijba->kc", ov, a2, t2)  # d15_ov
    He += 1./2 * e("ib,jkac,jkcb->ia", ov, a2, t2)  # d16_ov
    HHee += 1./2 * e("badc,ijba->ijdc", eris.vvvv, a2)  # d17_oovv
    He -= 1./2 * e("cbad,jicb,jd->ia", eris.vvvv, a2, t1)  # d18_ov
    He += 1./2 * e("bacj,ijba->ic", eris.vvvo, a2)  # d19_ov
    HHee -= p("ab..", e("jacb,ia->ijcb", eris.ovvv, a1))  # d20_oovv
    He -= e("jcab,ic,jb->ia", eris.ovvv, a1, t1)  # d21_ov                                    # + sign ?
    He -= e("jacb,ia,ic->jb", eris.ovvv, a1, t1)  # d22_ov
    HHee -= p("..ab", p("ab..", e("icda,kjcb,kd->ijab", eris.ovvv, a2, t1)))  # d23_oovv
    He += 1./2 * e("kbcd,ijba,ic,jd->ka", eris.ovvv, a2, t1, t1)  # d24_ov
    He -= e("kbad,ijcb,ia,kc->jd", eris.ovvv, a2, t1, t1)  # d25_ov
    He += e("jdbc,kida,jb,kc->ia", eris.ovvv, a2, t1, t1)  # d26_ov
    HHee -= e("ibdc,jkab,ia->jkdc", eris.ovvv, a2, t1)  # d27_oovv
    HHee += p("..ab", e("ibac,jkbd,ia->jkcd", eris.ovvv, a2, t1))  # d28_oovv
    He += 1./2 * e("icba,jkdc,jkbd->ia", eris.ovvv, a2, t2)  # d29_ov
    He += 1./4 * e("ibdc,jkba,jkdc->ia", eris.ovvv, a2, t2)  # d30_ov
    He += e("iabc,jkda,ijbd->kc", eris.ovvv, a2, t2)  # d31_ov
    He += 1./2 * e("jcba,ikcd,jiba->kd", eris.ovvv, a2, t2)  # d32_ov
    He += e("jabi,ia->jb", eris.ovvo, a1)  # d33_ov
    HHee += p("..ab", p("ab..", e("kaci,ijab->kjcb", eris.ovvo, a2)))  # d34_oovv
    He -= e("jcak,kicb,ia->jb", eris.ovvo, a2, t1)  # d35_ov
    He -= e("kbci,ijba,ka->jc", eris.ovvo, a2, t1)  # d36_ov
    He += e("jabi,ikac,jb->kc", eris.ovvo, a2, t1)  # d37_ov
    He += 1./2 * e("kaij,ijab->kb", eris.ovoo, a2)  # d38_ov
    HHee += e("ijba->ijba", eris.oovv)  # d39_oovv
    He += e("jiba,jb->ia", eris.oovv, t1)  # d40_ov
    HHee += p("..ab", e("ijba,kc,kb->ijac", eris.oovv, a1, t1))  # d41_oovv
    He -= e("ijab,kc,ka,ic->jb", eris.oovv, a1, t1, t1)  # d42_ov
    He -= e("ijab,kc,ka,jb->ic", eris.oovv, a1, t1, t1)  # d43_ov
    HHee += p("ab..", e("kiba,jc,kc->ijba", eris.oovv, a1, t1))  # d44_oovv
    He -= e("jkba,ic,jb,kc->ia", eris.oovv, a1, t1, t1)  # d45_ov
    HHee += p("..ab", p("ab..", e("ijab,kc,ia->jkbc", eris.oovv, a1, t1)))  # d46_oovv
    He += e("ikac,jb,ijab->kc", eris.oovv, a1, t2)  # d47_ov
    He -= 1./2 * e("ijba,kc,ikba->jc", eris.oovv, a1, t2)  # d48_ov
    He -= 1./2 * e("ijab,kc,ijac->kb", eris.oovv, a1, t2)  # d49_ov
    HHee -= p("ab..", e("ijab,lkdc,la,jb->ikdc", eris.oovv, a2, t1, t1))  # d50_oovv
    He -= e("jiab,klcd,jc,ka,ib->ld", eris.oovv, a2, t1, t1, t1)  # d51_ov
    HHee += p("..ab", p("ab..", e("jiab,klcd,ka,ic->jlbd", eris.oovv, a2, t1, t1)))  # d52_oovv
    He += 1./2 * e("ijcb,lkad,ia,jd,lc->kb", eris.oovv, a2, t1, t1, t1)  # d53_ov
    He -= 1./2 * e("jidc,lkab,ia,kc,ld->jb", eris.oovv, a2, t1, t1, t1)  # d54_ov
    HHee += 1./2 * e("klad,ijcb,ia,jd->klcb", eris.oovv, a2, t1, t1)  # d55_oovv
    He += 1./2 * e("ijab,lkdc,la,ijbd->kc", eris.oovv, a2, t1, t2)  # d56_ov
    He += 1./4 * e("klab,ijdc,ia,kldc->jb", eris.oovv, a2, t1, t2)  # d57_ov
    He += e("klcd,ijba,ic,ljdb->ka", eris.oovv, a2, t1, t2)  # d58_ov
    He -= 1./2 * e("liba,jkdc,jb,lkdc->ia", eris.oovv, a2, t1, t2)  # d59_ov
    HHee += p("..ab", e("ildc,jkab,ia,ld->jkcb", eris.oovv, a2, t1, t1))  # d60_oovv
    HHee += 1./2 * e("ildc,jkab,ia,lb->jkdc", eris.oovv, a2, t1, t1)  # d61_oovv
    He += 1./2 * e("lkdc,jiab,la,kjdc->ib", eris.oovv, a2, t1, t2)  # d62_ov
    He += e("lkdc,jiab,la,kjcb->id", eris.oovv, a2, t1, t2)  # d63_ov
    He += 1./4 * e("ijdc,klab,ia,kldc->jb", eris.oovv, a2, t1, t2)  # d64_ov
    He += 1./2 * e("ildc,jkab,ia,jkcb->ld", eris.oovv, a2, t1, t2)  # d65_ov
    He += e("lida,jkbc,ld,ijab->kc", eris.oovv, a2, t1, t2)  # d66_ov
    He += 1./2 * e("lidc,jkba,ld,ijba->kc", eris.oovv, a2, t1, t2)  # d67_ov
    He += 1./2 * e("lidb,jkca,ld,jkbc->ia", eris.oovv, a2, t1, t2)  # d68_ov
    HHee -= 1./2 * p("..ab", e("ijab,kldc,klad->ijbc", eris.oovv, a2, t2))  # d69_oovv
    HHee += 1./4 * e("klba,ijdc,ijba->kldc", eris.oovv, a2, t2)  # d70_oovv
    HHee -= 1./2 * p("ab..", e("ijba,lkdc,ildc->jkba", eris.oovv, a2, t2))  # d71_oovv
    HHee += p("..ab", p("ab..", e("ijab,klcd,ikac->jlbd", eris.oovv, a2, t2)))  # d72_oovv
    HHee += 1./2 * p("ab..", e("ildc,kjba,lkdc->ijba", eris.oovv, a2, t2))  # d73_oovv
    HHee += 1./4 * e("ijba,kldc,ijdc->klba", eris.oovv, a2, t2)  # d74_oovv
    HHee -= 1./2 * p("..ab", e("ijac,klbd,ijab->klcd", eris.oovv, a2, t2))  # d75_oovv
    HHee += p("..ab", e("jkbi,ia->jkab", eris.oovo, a1))  # d76_oovv
    He -= e("ikaj,jb,ia->kb", eris.oovo, a1, t1)  # d77_ov
    He -= e("ijak,kb,jb->ia", eris.oovo, a1, t1)  # d78_ov
    HHee += e("klci,jiba,jc->klba", eris.oovo, a2, t1)  # d79_oovv
    He += e("klcj,ijab,ka,ic->lb", eris.oovo, a2, t1, t1)  # d80_ov
    HHee += p("..ab", p("ab..", e("ijbl,lkac,ia->jkbc", eris.oovo, a2, t1)))  # d81_oovv
    He -= 1./2 * e("ijak,klbc,ib,jc->la", eris.oovo, a2, t1, t1)  # d82_ov
    He += e("jkai,ilcb,jc,ka->lb", eris.oovo, a2, t1, t1)  # d83_ov
    HHee -= p("ab..", e("kicl,ljba,kc->ijba", eris.oovo, a2, t1))  # d84_oovv
    He -= 1./2 * e("ilck,jkba,ijba->lc", eris.oovo, a2, t2)  # d85_ov
    He -= e("ikaj,ljbc,ilab->kc", eris.oovo, a2, t2)  # d86_ov
    He -= 1./4 * e("jkal,licb,jkcb->ia", eris.oovo, a2, t2)  # d87_ov
    He -= 1./2 * e("klbj,jica,klbc->ia", eris.oovo, a2, t2)  # d88_ov
    HHee += 1./2 * e("ijkl,klba->ijba", eris.oooo, a2)  # d89_oovv
    He -= 1./2 * e("iljk,jkba,lb->ia", eris.oooo, a2, t1)  # d90_ov
    return He, HHee

def La2eq(t2, a2, eris, fsp=None):

    nocc = t2.shape[0]
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    vv = f[nocc:,nocc:].copy()
    oo = f[:nocc,:nocc].copy()

    HHee = 0
    HHee += p("..ab", e("ba,ijbc->ijac", vv, a2))  # d0_oovv
    HHee -= p("ab..", e("ij,jkba->ikba", oo, a2))  # d1_oovv
    HHee += 1./2 * e("dcba,ijdc->ijba", eris.vvvv, a2)  # d2_oovv
    HHee += p("..ab", p("ab..", e("kbcj,jiba->kica", eris.ovvo, a2)))  # d3_oovv
    HHee += e("ijba->ijba", eris.oovv)  # d4_oovv
    HHee -= 1./2 * p("..ab", e("ijab,klcd,klac->ijbd", eris.oovv, a2, t2))  # d5_oovv
    HHee += 1./4 * e("ijba,kldc,klba->ijdc", eris.oovv, a2, t2)  # d6_oovv
    HHee -= 1./2 * p("ab..", e("ikdc,jlba,ijba->kldc", eris.oovv, a2, t2))  # d7_oovv
    HHee += p("..ab", p("ab..", e("klcd,jiba,jkbc->lida", eris.oovv, a2, t2)))  # d8_oovv
    HHee -= 1./2 * p("ab..", e("kidc,ljba,kldc->ijba", eris.oovv, a2, t2))  # d9_oovv
    HHee += 1./4 * e("ijba,kldc,ijdc->klba", eris.oovv, a2, t2)  # d10_oovv
    HHee -= 1./2 * p("..ab", e("ijab,kldc,ijad->klbc", eris.oovv, a2, t2))  # d11_oovv
    HHee += 1./2 * e("ijkl,klba->ijba", eris.oooo, a2)  # d12_oovv
    return HHee

# R equation
# -----------------

# L equation
# -----------------

##########
# CCSDT
##########

# Energy
# -----------------

# T equations
# -----------------
def eq_gs_sdt(t1, t2, t3, eris, fsp=None):
    
    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    vv = f[nocc:,nocc:].copy()
    oo = f[:nocc,:nocc].copy()
    vo = f[nocc:,:nocc].copy()
    ov = f[:nocc,nocc:].copy()
    
    hE = hhEE = hhhEEE = 0
    hE += e("ba,ia->ib", vv, t1)  # d0_ov
    hhEE += p("..ab", e("ac,ijcb->ijab", vv, t2))  # d1_oovv
    hhhEEE += p("...abb", e("ba,jkiadc->jkibdc", vv, t3))  # d2_ooovvv
    hE -= e("ij,ia->ja", oo, t1)  # d3_ov
    hhEE -= p("ab..", e("ij,ikba->jkba", oo, t2))  # d4_oovv
    hhhEEE -= p("abb...", e("il,ikjcba->lkjcba", oo, t3))  # d5_ooovvv
    hE += e("ai->ia", vo)  # d6_ov
    hE -= e("ia,ja,ib->jb", ov, t1, t1)  # d7_ov
    hhEE -= p("ab..", e("kc,ic,kjba->ijba", ov, t1, t2))  # d8_oovv
    hhhEEE -= p("abb...", e("ld,kd,ljicba->kjicba", ov, t1, t3))  # d9_ooovvv
    hhEE -= p("..ab", e("kc,ka,ijcb->ijab", ov, t1, t2))  # d10_oovv
    hhhEEE -= p("...abb", e("la,ld,jkiacb->jkidcb", ov, t1, t3))  # d11_ooovvv
    hhhEEE -= p("aab...", p("...abb", e("id,ijba,klcd->kljcba", ov, t2, t2)))  # d12_ooovvv
    hE += e("ia,ijab->jb", ov, t2)  # d13_ov
    hhEE += e("ia,ikjacb->kjcb", ov, t3)  # d14_oovv
    hhEE += 1./2 * p("ab..", e("cbad,ia,jd->ijcb", eris.vvvv, t1, t1))  # d15_oovv
    hhhEEE += p("abb...", p("...aab", e("dcea,ke,ijab->kijdcb", eris.vvvv, t1, t2)))  # d16_ooovvv
    hhEE += 1./2 * e("dcba,ijba->ijdc", eris.vvvv, t2)  # d17_oovv
    hhhEEE += 1./2 * p("...abb", e("badc,jkidce->jkieba", eris.vvvv, t3))  # d18_ooovvv
    hhEE += p("ab..", e("cbaj,ia->ijcb", eris.vvvo, t1))  # d19_oovv
    hhhEEE -= p("aab...", p("...abb", e("dcak,ijab->ijkbdc", eris.vvvo, t2)))  # d20_ooovvv
    hhEE += e("baij->ijba", eris.vvoo)  # d21_oovv
    hhEE -= 1./2 * p("ab..", p("..ab", e("kcba,ia,jb,kd->ijcd", eris.ovvv, t1, t1, t1)))  # d22_oovv
    hhhEEE += 1./2 * p("abc...", p("...abb", e("kcba,ia,jb,kled->iljced", eris.ovvv, t1, t1, t2)))  # d23_ooovvv
    hhhEEE -= p("aab...", p("...abc", e("ibca,ja,id,klce->kljbde", eris.ovvv, t1, t1, t2)))  # d24_ooovvv
    hE -= e("ibca,ia,jc->jb", eris.ovvv, t1, t1)  # d25_ov
    hhEE += p("ab..", p("..ab", e("jcdb,kd,jiba->ikca", eris.ovvv, t1, t2)))  # d26_oovv
    hhhEEE -= p("aab...", p("...abb", e("icba,lb,ijkaed->jklced", eris.ovvv, t1, t3)))  # d27_ooovvv
    hhEE += 1./2 * p("..ab", e("kadc,kb,ijdc->ijab", eris.ovvv, t1, t2))  # d28_oovv
    hhhEEE -= 1./2 * p("...abc", e("ibed,ia,kljedc->kljbca", eris.ovvv, t1, t3))  # d29_ooovvv
    hhEE -= p("..ab", e("icba,ia,jkbd->jkcd", eris.ovvv, t1, t2))  # d30_oovv
    hhhEEE -= p("...abb", e("iabc,ic,kljbed->kljaed", eris.ovvv, t1, t3))  # d31_ooovvv
    hhhEEE -= p("abb...", p("...abc", e("ibac,jkcd,ilae->ljkbed", eris.ovvv, t2, t2)))  # d32_ooovvv
    hhhEEE += 1./2 * p("abb...", p("...abb", e("iacb,iled,kjcb->lkjaed", eris.ovvv, t2, t2)))  # d33_ooovvv
    hE += 1./2 * e("icba,ijba->jc", eris.ovvv, t2)  # d34_ov
    hhEE += 1./2 * p("..ab", e("kadc,kijdcb->ijab", eris.ovvv, t3))  # d35_oovv
    hhEE += p("ab..", p("..ab", e("kcaj,ia,kb->ijcb", eris.ovvo, t1, t1)))  # d36_oovv
    hhhEEE += p("abc...", p("...abb", e("jbai,ka,jldc->kilbdc", eris.ovvo, t1, t2)))  # d37_ooovvv
    hhhEEE += p("abb...", p("...abc", e("icbj,ia,lkbd->jlkacd", eris.ovvo, t1, t2)))  # d38_ooovvv
    hE += e("ibaj,ia->jb", eris.ovvo, t1)  # d39_ov
    hhEE += p("ab..", p("..ab", e("kbcj,kica->ijab", eris.ovvo, t2)))  # d40_oovv
    hhhEEE += p("abb...", p("...abb", e("jabi,jlkbdc->ilkadc", eris.ovvo, t3)))  # d41_ooovvv
    hhEE += p("..ab", e("iajk,ib->jkab", eris.ovoo, t1))  # d42_oovv
    hhhEEE += p("aab...", p("...abb", e("ickl,ijba->kljcba", eris.ovoo, t2)))  # d43_ooovvv
    hhEE += 1./4 * p("ab..", p("..ab", e("lidc,ia,jc,kd,lb->kjba", eris.oovv, t1, t1, t1, t1)))  # d44_oovv
    hhhEEE -= 1./2 * p("abc...", p("...aab", e("ijbc,ia,kb,lc,jmed->mlkeda", eris.oovv, t1, t1, t1, t2)))  # d45_ooovvv
    hhEE -= 1./4 * p("ab..", e("ijba,kb,la,ijdc->lkdc", eris.oovv, t1, t1, t2))  # d46_oovv
    hhhEEE += 1./4 * p("abc...", e("lmde,kd,je,lmicba->ikjcba", eris.oovv, t1, t1, t3))  # d47_ooovvv
    hhhEEE -= 1./2 * p("aab...", p("...abc", e("mled,id,la,mc,jkeb->jkibca", eris.oovv, t1, t1, t1, t2)))  # d48_ooovvv
    hE -= e("jiab,ka,jc,ib->kc", eris.oovv, t1, t1, t1)  # d49_ov
    hhEE -= p("ab..", p("..ab", e("klcd,lb,jd,kica->ijab", eris.oovv, t1, t1, t2)))  # d50_oovv
    hhhEEE -= p("aab...", p("...aab", e("mled,ma,ke,lijdcb->ijkcba", eris.oovv, t1, t1, t3)))  # d51_ooovvv
    hhEE -= p("ab..", e("jlcd,lc,kd,jiba->ikba", eris.oovv, t1, t1, t2))  # d52_oovv
    hhhEEE -= p("aab...", e("miba,ia,jb,mkledc->kljedc", eris.oovv, t1, t1, t3))  # d53_ooovvv
    hhhEEE -= 1./2 * p("abb...", p("...aab", e("jkba,ia,mlbe,jkdc->imldce", eris.oovv, t1, t2, t2)))  # d54_ooovvv
    hhhEEE -= p("abc...", p("...abb", e("lmde,ie,ljda,mkcb->jikacb", eris.oovv, t1, t2, t2)))  # d55_ooovvv
    hE -= 1./2 * e("jkba,ia,jkbc->ic", eris.oovv, t1, t2)  # d56_ov
    hhEE -= 1./2 * p("ab..", e("klcd,id,kljcba->ijba", eris.oovv, t1, t3))  # d57_oovv
    hhEE += 1./4 * p("..ab", e("ijba,jc,id,klba->kldc", eris.oovv, t1, t1, t2))  # d58_oovv
    hhhEEE -= 1./4 * p("...abc", e("mled,lc,ma,jkibed->jkibca", eris.oovv, t1, t1, t3))  # d59_ooovvv
    hhEE -= p("..ab", e("klcd,ld,ka,ijcb->ijab", eris.oovv, t1, t1, t2))  # d60_oovv
    hhhEEE += p("...abb", e("jiab,ia,je,lmkbdc->lmkedc", eris.oovv, t1, t1, t3))  # d61_ooovvv
    hhhEEE += p("abb...", p("...abc", e("klda,kc,ijab,lmde->mijecb", eris.oovv, t1, t2, t2)))  # d62_ooovvv
    hhhEEE += 1./2 * p("abb...", p("...aab", e("ilcb,ia,jkcb,lmed->mjkeda", eris.oovv, t1, t2, t2)))  # d63_ooovvv
    hE += 1./2 * e("ikcb,ia,kjcb->ja", eris.oovv, t1, t2)  # d64_ov
    hhEE -= 1./2 * p("..ab", e("ilba,ld,ikjbac->kjdc", eris.oovv, t1, t3))  # d65_oovv
    hhhEEE += p("abb...", p("...aab", e("lmae,me,ijab,lkdc->kijdcb", eris.oovv, t1, t2, t2)))  # d66_ooovvv
    hE += e("ikac,ia,kjcb->jb", eris.oovv, t1, t2)  # d67_ov
    hhEE += e("ijab,jb,ikladc->kldc", eris.oovv, t1, t3)  # d68_oovv
    hhEE += 1./2 * p("..ab", e("klac,ijab,klcd->ijdb", eris.oovv, t2, t2))  # d69_oovv
    hhhEEE += 1./2 * p("abb...", p("...aab", e("kjce,mled,ikjbac->imlbad", eris.oovv, t2, t3)))  # d70_ooovvv
    hhEE += 1./4 * e("ijba,ijdc,klba->kldc", eris.oovv, t2, t2)  # d71_oovv
    hhhEEE += 1./4 * p("abb...", e("ijed,mled,ijkcba->kmlcba", eris.oovv, t2, t3))  # d72_ooovvv
    hhEE -= 1./2 * p("ab..", e("ikba,ijba,kldc->jldc", eris.oovv, t2, t2))  # d73_oovv
    hhhEEE += 1./2 * p("aab...", p("...abb", e("lmed,mkcb,lijeda->ijkacb", eris.oovv, t2, t3)))  # d74_ooovvv
    hhEE += 1./2 * p("ab..", p("..ab", e("ijab,jkbc,ilad->klcd", eris.oovv, t2, t2)))  # d75_oovv
    hhhEEE += p("aab...", p("...aab", e("mled,lkdc,mijeba->ijkbac", eris.oovv, t2, t3)))  # d76_ooovvv
    hhhEEE += 1./2 * p("aab...", e("ijba,jmba,ikledc->klmedc", eris.oovv, t2, t3))  # d77_ooovvv
    hhhEEE += 1./4 * p("...abb", e("lmcb,lmed,jkicba->jkiaed", eris.oovv, t2, t3))  # d78_ooovvv
    hhhEEE += 1./2 * p("...aab", e("ijba,ijac,lmkbed->lmkedc", eris.oovv, t2, t3))  # d79_ooovvv
    hE += 1./4 * e("ijba,ijkbac->kc", eris.oovv, t3)  # d80_ov
    hhEE += 1./2 * p("ab..", p("..ab", e("kiaj,ib,la,kc->ljcb", eris.oovo, t1, t1, t1)))  # d81_oovv
    hhhEEE -= p("abc...", p("...aab", e("mldi,la,jd,kmcb->jkicba", eris.oovo, t1, t1, t2)))  # d82_ooovvv
    hhEE += 1./2 * p("ab..", e("klcj,ic,klba->ijba", eris.oovo, t1, t2))  # d83_oovv
    hhhEEE += 1./2 * p("abc...", e("lmak,ia,lmjdcb->ikjdcb", eris.oovo, t1, t3))  # d84_ooovvv
    hhhEEE += 1./2 * p("abb...", p("...abc", e("ijak,ib,jc,mlad->kmlbdc", eris.oovo, t1, t1, t2)))  # d85_ooovvv
    hE += e("ijak,ib,ja->kb", eris.oovo, t1, t1)  # d86_ov
    hhEE += p("ab..", p("..ab", e("ikbj,ia,klbc->jlac", eris.oovo, t1, t2)))  # d87_oovv
    hhhEEE += p("abb...", p("...abb", e("ikbj,ia,kmlbdc->jmladc", eris.oovo, t1, t3)))  # d88_ooovvv
    hhEE -= p("ab..", e("ijal,ia,jkcb->lkcb", eris.oovo, t1, t2))  # d89_oovv
    hhhEEE -= p("abb...", e("lmdk,ld,mjicba->kjicba", eris.oovo, t1, t3))  # d90_ooovvv
    hhhEEE += 1./2 * p("aab...", p("...abb", e("lmak,ijba,lmdc->ijkbdc", eris.oovo, t2, t2)))  # d91_ooovvv
    hhhEEE -= p("abc...", p("...aab", e("lmdk,liba,mjdc->ikjbac", eris.oovo, t2, t2)))  # d92_ooovvv
    hE -= 1./2 * e("ijak,ijab->kb", eris.oovo, t2)  # d93_ov
    hhEE += 1./2 * p("ab..", e("ijal,ijkacb->klcb", eris.oovo, t3))  # d94_oovv
    hhEE += 1./2 * p("..ab", e("iljk,ia,lb->jkab", eris.oooo, t1, t1))  # d95_oovv
    hhhEEE += p("aab...", p("...aab", e("iljk,ia,lmcb->jkmcba", eris.oooo, t1, t2)))  # d96_ooovvv
    hhEE += 1./2 * e("ijkl,ijba->klba", eris.oooo, t2)  # d97_oovv
    hhhEEE += 1./2 * p("aab...", e("ijkl,ijmcba->klmcba", eris.oooo, t3))  # d98_ooovvv
    return hE, hhEE, hhhEEE

# Lambda equations
# ------------------------
def eq_lambda_sdt(t1, t2, t3, a1, a2, a3, eris, fsp=None):
    
    nocc,nvir = t1.shape
    if fsp is None:
        f = eris.fock.copy()
    else:
        f = fsp

    vv = f[nocc:,nocc:].copy()
    oo = f[:nocc,:nocc].copy()
    vo = f[nocc:,:nocc].copy()
    ov = f[:nocc,nocc:].copy()

    He = HHee = HHHeee = 0
    He += e("ba,ib->ia", vv, a1)  # d0_ov
    HHee += p("..ab", e("ca,ijcb->ijab", vv, a2))  # d1_oovv
    He += e("ba,jibc,ja->ic", vv, a2, t1)  # d2_ov
    HHHeee += p("...abb", e("da,jkidcb->jkiacb", vv, a3))  # d3_ooovvv
    HHee += e("ab,ijkadc,ib->jkdc", vv, a3, t1)  # d4_oovv
    He += 1./2 * e("db,ijkdac,ijba->kc", vv, a3, t2)  # d5_ov
    He -= e("ji,ia->ja", oo, a1)  # d6_ov
    HHee -= p("ab..", e("ki,ijba->kjba", oo, a2))  # d7_oovv
    He -= e("ij,jkab,ia->kb", oo, a2, t1)  # d8_ov
    HHHeee -= p("abb...", e("lk,kjicba->ljicba", oo, a3))  # d9_ooovvv
    HHee -= e("li,ikjacb,la->kjcb", oo, a3, t1)  # d10_oovv
    He -= 1./2 * e("jl,likbac,jiba->kc", oo, a3, t2)  # d11_ov
    He += e("bj,ijab->ia", vo, a2)  # d12_ov
    HHee += e("ai,ijkacb->jkcb", vo, a3)  # d13_oovv
    He += e("ia->ia", ov)  # d14_ov
    HHee += p("..ab", p("ab..", e("jb,ia->jiba", ov, a1)))  # d15_oovv
    He -= e("ib,ja,jb->ia", ov, a1, t1)  # d16_ov
    He -= e("ja,ib,jb->ia", ov, a1, t1)  # d17_ov
    HHHeee += p("...abb", p("abb...", e("ia,jkcb->ijkacb", ov, a2)))  # d18_ooovvv
    HHee -= p("ab..", e("ic,kjba,kc->ijba", ov, a2, t1))  # d19_oovv
    He += e("ja,ikcb,ia,jb->kc", ov, a2, t1, t1)  # d20_ov
    HHee += p("..ab", e("ka,ijbc,kc->ijab", ov, a2, t1))  # d21_oovv
    He += 1./2 * e("ib,jkca,jkbc->ia", ov, a2, t2)  # d22_ov
    He += 1./2 * e("kc,ijba,kiba->jc", ov, a2, t2)  # d23_ov
    He += e("kc,ijab,kjcb->ia", ov, a2, t2)  # d24_ov
    HHHeee -= p("abb...", e("ja,ilkdcb,ia->jlkdcb", ov, a3, t1))  # d25_ooovvv
    HHee -= e("ia,jkldcb,ja,id->klcb", ov, a3, t1, t1)  # d26_oovv
    He -= 1./2 * e("ia,ljkdcb,ka,ildc->jb", ov, a3, t1, t2)  # d27_ov
    HHHeee -= p("...abb", e("ib,kljdca,ia->kljbdc", ov, a3, t1))  # d28_ooovvv
    He += 1./2 * e("jb,klidca,jc,klbd->ia", ov, a3, t1, t2)  # d29_ov
    HHee += 1./2 * p("ab..", e("ka,ijlbdc,ijab->kldc", ov, a3, t2))  # d30_oovv
    HHee += 1./2 * p("..ab", e("ic,jklbad,ijba->klcd", ov, a3, t2))  # d31_oovv
    HHee += e("ia,jklcbd,ilad->jkcb", ov, a3, t2)  # d32_oovv
    He -= 1./12 * e("ib,kljadc,kljbdc->ia", ov, a3, t3)  # d33_ov
    He -= 1./12 * e("ia,kljdcb,ikldcb->ja", ov, a3, t3)  # d34_ov
    He += 1./4 * e("ia,jlkbdc,ilkadc->jb", ov, a3, t3)  # d35_ov
    HHee += 1./2 * e("dcba,ijdc->ijba", eris.vvvv, a2)  # d36_oovv
    He -= 1./2 * e("cbda,ijcb,jd->ia", eris.vvvv, a2, t1)  # d37_ov
    HHHeee += 1./2 * p("...aab", e("badc,jkibae->jkidce", eris.vvvv, a3))  # d38_ooovvv
    HHee += 1./2 * p("..ab", e("cbad,ijkcbe,ia->jkde", eris.vvvv, a3, t1))  # d39_oovv
    He += 1./4 * e("dcbe,jkidca,jb,ke->ia", eris.vvvv, a3, t1, t1)  # d40_ov
    He -= 1./4 * e("edac,ijkbed,ijab->kc", eris.vvvv, a3, t2)  # d41_ov
    He += 1./8 * e("edba,ijkedc,ijba->kc", eris.vvvv, a3, t2)  # d42_ov
    He -= 1./2 * e("baci,ijba->jc", eris.vvvo, a2)  # d43_ov
    HHee += 1./2 * p("..ab", e("badi,ikjbac->kjcd", eris.vvvo, a3))  # d44_oovv
    He -= 1./2 * e("badi,ijkbac,jd->kc", eris.vvvo, a3, t1)  # d45_ov
    He += 1./4 * e("baij,ijkbac->kc", eris.vvoo, a3)  # d46_ov
    HHee += p("ab..", e("icba,jc->ijba", eris.ovvv, a1))  # d47_oovv
    He -= e("ibca,jb,jc->ia", eris.ovvv, a1, t1)  # d48_ov
    He += e("ibac,jb,ia->jc", eris.ovvv, a1, t1)  # d49_ov
    HHHeee += p("...abb", p("aab...", e("icba,jkcd->jkidba", eris.ovvv, a2)))  # d50_ooovvv
    HHee += p("..ab", e("kadc,ijab,kc->ijbd", eris.ovvv, a2, t1))  # d51_oovv
    He -= e("jcab,ikcd,ia,jb->kd", eris.ovvv, a2, t1, t1)  # d52_ov
    HHee += e("kadc,ijab,kb->ijdc", eris.ovvv, a2, t1)  # d53_oovv
    He += e("kbda,ijcb,jd,kc->ia", eris.ovvv, a2, t1, t1)  # d54_ov
    HHee -= p("..ab", p("ab..", e("kbdc,ijab,jd->ikac", eris.ovvv, a2, t1)))  # d55_oovv
    He -= 1./2 * e("iacb,jkad,jb,kc->id", eris.ovvv, a2, t1, t1)  # d56_ov
    He += 1./2 * e("icba,kjcd,jiba->kd", eris.ovvv, a2, t2)  # d57_ov
    He += e("jcbd,ikac,ijab->kd", eris.ovvv, a2, t2)  # d58_ov
    He += 1./4 * e("iacb,jkad,jkcb->id", eris.ovvv, a2, t2)  # d59_ov
    He += 1./2 * e("kadc,ijab,ijbd->kc", eris.ovvv, a2, t2)  # d60_ov
    HHHeee -= p("...aab", e("lade,jkiacb,le->jkicbd", eris.ovvv, a3, t1))  # d61_ooovvv
    HHee -= e("lcde,ijkbac,le,kd->ijba", eris.ovvv, a3, t1, t1)  # d62_oovv
    He += 1./2 * e("lbed,jkicba,ld,jkec->ia", eris.ovvv, a3, t1, t2)  # d63_ov
    HHHeee -= p("...abb", e("iedc,kljaeb,ia->kljbdc", eris.ovvv, a3, t1))  # d64_ooovvv
    HHee += p("..ab", e("ldcb,kijaed,kc,le->ijab", eris.ovvv, a3, t1, t1))  # d65_oovv
    He -= 1./2 * e("kdab,iljced,ia,jb,kc->le", eris.ovvv, a3, t1, t1, t1)  # d66_ov
    He += 1./4 * e("ibdc,jlkeba,ia,lkdc->je", eris.ovvv, a3, t1, t2)  # d67_ov
    He -= 1./2 * e("iacb,jlkaed,id,lkec->jb", eris.ovvv, a3, t1, t2)  # d68_ov
    HHHeee -= p("...aab", p("aab...", e("lced,ijkbac,ke->ijlbad", eris.ovvv, a3, t1)))  # d69_ooovvv
    HHee += 1./2 * p("ab..", e("leab,ikjdce,ia,jb->kldc", eris.ovvv, a3, t1, t1))  # d70_oovv
    He += e("jebc,ilkade,kc,ijab->ld", eris.ovvv, a3, t1, t2)  # d71_ov
    He += 1./2 * e("lbae,ikjbdc,je,kldc->ia", eris.ovvv, a3, t1, t2)  # d72_ov
    He += 1./2 * e("iabc,kljade,jc,kleb->id", eris.ovvv, a3, t1, t2)  # d73_ov
    HHee -= 1./2 * e("kedc,ijlbae,lkdc->ijba", eris.ovvv, a3, t2)  # d74_oovv
    HHee += p("..ab", e("lbed,ijkabc,klce->ijad", eris.ovvv, a3, t2))  # d75_oovv
    HHee -= 1./2 * e("lced,ikjbac,ilba->kjed", eris.ovvv, a3, t2)  # d76_oovv
    HHee -= 1./4 * p("ab..", e("iacb,jlkaed,lkcb->jied", eris.ovvv, a3, t2))  # d77_oovv
    HHee += 1./2 * p("..ab", p("ab..", e("lbed,jkibca,jkce->ilad", eris.ovvv, a3, t2)))  # d78_oovv
    He -= 1./4 * e("lbed,ikjbac,kjlced->ia", eris.ovvv, a3, t3)  # d79_ov
    He += 1./4 * e("laed,ikjacb,kjlcbe->id", eris.ovvv, a3, t3)  # d80_ov
    He -= 1./12 * e("leba,jkidec,jkicba->ld", eris.ovvv, a3, t3)  # d81_ov
    He += 1./12 * e("lcde,jkibac,jkibae->ld", eris.ovvv, a3, t3)  # d82_ov
    He += e("ibaj,jb->ia", eris.ovvo, a1)  # d83_ov
    HHee += p("..ab", p("ab..", e("ibaj,jkbc->ikac", eris.ovvo, a2)))  # d84_oovv
    He -= e("icbk,kjca,jb->ia", eris.ovvo, a2, t1)  # d85_ov
    He -= e("ibcj,jkba,ia->kc", eris.ovvo, a2, t1)  # d86_ov
    He += e("kbcj,jiba,kc->ia", eris.ovvo, a2, t1)  # d87_ov
    HHHeee += p("...abb", p("abb...", e("ibaj,jlkbdc->ilkadc", eris.ovvo, a3)))  # d88_ooovvv
    HHee += p("ab..", e("ldak,ikjdcb,ia->ljcb", eris.ovvo, a3, t1))  # d89_oovv
    He += e("icdl,kljcab,ia,kd->jb", eris.ovvo, a3, t1, t1)  # d90_ov
    HHee += p("..ab", e("idbl,ljkadc,ia->jkbc", eris.ovvo, a3, t1))  # d91_oovv
    HHee += e("lcdk,kjicba,ld->jiba", eris.ovvo, a3, t1)  # d92_oovv
    He += 1./2 * e("jabi,ikladc,klbd->jc", eris.ovvo, a3, t2)  # d93_ov
    He += 1./2 * e("jabi,ilkadc,jldc->kb", eris.ovvo, a3, t2)  # d94_ov
    He += e("ibaj,jlkbdc,ilad->kc", eris.ovvo, a3, t2)  # d95_ov
    He += 1./2 * e("kaij,ijab->kb", eris.ovoo, a2)  # d96_ov
    HHee += 1./2 * p("ab..", e("laij,ijkacb->lkcb", eris.ovoo, a3))  # d97_oovv
    He -= 1./2 * e("ickl,kljacb,ia->jb", eris.ovoo, a3, t1)  # d98_ov
    HHee += e("ijba->ijba", eris.oovv)  # d99_oovv
    He += e("ijab,jb->ia", eris.oovv, t1)  # d100_ov
    HHHeee += p("...aab", p("aab...", e("jkcb,ia->jkicba", eris.oovv, a1)))  # d101_ooovvv
    HHee -= p("..ab", e("ijac,kb,kc->ijab", eris.oovv, a1, t1))  # d102_oovv
    He -= e("jkbc,ia,ja,ib->kc", eris.oovv, a1, t1, t1)  # d103_ov
    He -= e("ijac,kb,ia,kc->jb", eris.oovv, a1, t1, t1)  # d104_ov
    HHee -= p("ab..", e("ikba,jc,kc->ijba", eris.oovv, a1, t1))  # d105_oovv
    He += e("ijba,kc,jb,ic->ka", eris.oovv, a1, t1, t1)  # d106_ov
    HHee += p("..ab", p("ab..", e("jiba,kc,jb->ikac", eris.oovv, a1, t1)))  # d107_oovv
    He += e("ijab,kc,ikac->jb", eris.oovv, a1, t2)  # d108_ov
    He -= 1./2 * e("ikba,jc,ijba->kc", eris.oovv, a1, t2)  # d109_ov
    He -= 1./2 * e("ijab,kc,ijac->kb", eris.oovv, a1, t2)  # d110_ov
    HHHeee += p("...abb", p("aab...", e("kldc,ijba,id->kljcba", eris.oovv, a2, t1)))  # d111_ooovvv
    HHee += 1./2 * e("ijcd,lkba,kd,lc->ijba", eris.oovv, a2, t1, t1)  # d112_oovv
    He += 1./2 * e("jiba,klcd,ka,ic,lb->jd", eris.oovv, a2, t1, t1, t1)  # d113_ov
    HHee -= p("..ab", p("ab..", e("lkcd,jiab,ka,jd->licb", eris.oovv, a2, t1, t1)))  # d114_oovv
    He += 1./2 * e("licd,kjab,ia,lb,kd->jc", eris.oovv, a2, t1, t1, t1)  # d115_ov
    He -= e("lkdc,ijab,kc,id,la->jb", eris.oovv, a2, t1, t1, t1)  # d116_ov
    HHee += p("ab..", e("ijba,lkdc,ia,lb->jkdc", eris.oovv, a2, t1, t1))  # d117_oovv
    He += 1./2 * e("jiab,kldc,la,jkdc->ib", eris.oovv, a2, t1, t2)  # d118_ov
    He -= e("ijab,klcd,lb,ikac->jd", eris.oovv, a2, t1, t2)  # d119_ov
    He += 1./4 * e("klab,ijdc,ia,kldc->jb", eris.oovv, a2, t1, t2)  # d120_ov
    He += 1./2 * e("kldc,jiba,jd,klcb->ia", eris.oovv, a2, t1, t2)  # d121_ov
    HHHeee += p("...aab", p("abb...", e("liba,kjdc,ld->ikjbac", eris.oovv, a2, t1)))  # d122_ooovvv
    HHee += 1./2 * e("jiba,kldc,jd,ic->klba", eris.oovv, a2, t1, t1)  # d123_oovv
    HHee -= p("..ab", e("ilad,jkcb,ia,lc->jkdb", eris.oovv, a2, t1, t1))  # d124_oovv
    He += 1./2 * e("ijab,klcd,id,klac->jb", eris.oovv, a2, t1, t2)  # d125_ov
    He += 1./4 * e("jiba,kldc,jd,klba->ic", eris.oovv, a2, t1, t2)  # d126_ov
    He += e("kldc,jiab,kb,ljda->ic", eris.oovv, a2, t1, t2)  # d127_ov
    He -= 1./2 * e("kldc,ijba,lb,kidc->ja", eris.oovv, a2, t1, t2)  # d128_ov
    HHHeee += p("...abb", p("abb...", e("ijab,lkdc,ia->jlkbdc", eris.oovv, a2, t1)))  # d129_ooovvv
    He -= 1./2 * e("lkcd,ijab,kc,ijda->lb", eris.oovv, a2, t1, t2)  # d130_ov
    He -= 1./2 * e("ijba,lkdc,jb,ildc->ka", eris.oovv, a2, t1, t2)  # d131_ov
    He += e("klcd,ijab,ld,kica->jb", eris.oovv, a2, t1, t2)  # d132_ov
    HHee -= 1./2 * p("..ab", e("klac,ijbd,ijab->klcd", eris.oovv, a2, t2))  # d133_oovv
    HHee += 1./4 * e("ijba,kldc,klba->ijdc", eris.oovv, a2, t2)  # d134_oovv
    HHee -= 1./2 * p("ab..", e("ikdc,jlba,ijba->kldc", eris.oovv, a2, t2))  # d135_oovv
    HHee += p("..ab", p("ab..", e("ijab,klcd,ikac->jlbd", eris.oovv, a2, t2)))  # d136_oovv
    HHee -= 1./2 * p("ab..", e("jiba,lkdc,jlba->ikdc", eris.oovv, a2, t2))  # d137_oovv
    HHee += 1./4 * e("ijba,kldc,ijdc->klba", eris.oovv, a2, t2)  # d138_oovv
    HHee -= 1./2 * p("..ab", e("klca,ijdb,klcd->ijab", eris.oovv, a2, t2))  # d139_oovv
    He += 1./4 * e("jiba,lkdc,jlkbdc->ia", eris.oovv, a2, t3)  # d140_ov
    He += 1./4 * e("ilba,kjcd,ikjbac->ld", eris.oovv, a2, t3)  # d141_ov
    He += 1./4 * e("ijab,kldc,ijkadc->lb", eris.oovv, a2, t3)  # d142_ov
    He += 1./4 * e("kjcb,ilad,ikjacb->ld", eris.oovv, a2, t3)  # d143_ov
    HHHeee -= 1./2 * p("aab...", e("jked,limcba,md,le->jkicba", eris.oovv, a3, t1, t1))  # d144_ooovvv
    HHee += 1./2 * p("ab..", e("mlde,ikjcba,id,ke,mc->ljba", eris.oovv, a3, t1, t1, t1))  # d145_oovv
    He += 1./4 * e("lmed,kijcab,jd,ke,lc,mb->ia", eris.oovv, a3, t1, t1, t1, t1)  # d146_ov
    He += 1./4 * e("imed,ljkbac,le,kd,ijba->mc", eris.oovv, a3, t1, t1, t2)  # d147_ov
    He -= 1./8 * e("ijab,kmldce,ka,lb,ijdc->me", eris.oovv, a3, t1, t1, t2)  # d148_ov
    HHHeee -= p("...abb", p("abb...", e("lkec,mjidba,ld,me->kjicba", eris.oovv, a3, t1, t1)))  # d149_ooovvv
    HHee -= 1./2 * p("..ab", e("imbe,ljkadc,ia,le,md->jkbc", eris.oovv, a3, t1, t1, t1))  # d150_oovv
    HHee += e("kjba,imlced,ia,jb,kc->mled", eris.oovv, a3, t1, t1, t1)  # d151_oovv
    He += 1./2 * e("klbd,ijmcae,kc,md,ijba->le", eris.oovv, a3, t1, t1, t2)  # d152_ov
    He += 1./2 * e("mlae,kijbdc,je,mb,lkdc->ia", eris.oovv, a3, t1, t1, t2)  # d153_ov
    He += e("mlde,ikjbac,mb,je,lida->kc", eris.oovv, a3, t1, t1, t2)  # d154_ov
    HHHeee -= p("abb...", e("lide,kjmcba,ld,me->ikjcba", eris.oovv, a3, t1, t1))  # d155_ooovvv
    He += 1./2 * e("ikac,mljedb,ia,lc,kmed->jb", eris.oovv, a3, t1, t1, t2)  # d156_ov
    HHee += 1./2 * e("ijba,klmdce,kb,lmae->ijdc", eris.oovv, a3, t1, t2)  # d157_oovv
    HHee += 1./2 * p("..ab", p("ab..", e("ikec,jmlbad,me,ijba->klcd", eris.oovv, a3, t1, t2)))  # d158_oovv
    HHee -= p("ab..", e("mlde,ikjacb,ke,mida->ljcb", eris.oovv, a3, t1, t2))  # d159_oovv
    HHee += 1./4 * p("..ab", e("jkad,ilmecb,ia,jkcb->lmde", eris.oovv, a3, t1, t2))  # d160_oovv
    HHee += 1./2 * e("ijba,mkldce,mb,ijae->kldc", eris.oovv, a3, t1, t2)  # d161_oovv
    He += 1./12 * e("mled,ijkcba,kd,ijmcba->le", eris.oovv, a3, t1, t3)  # d162_ov
    He -= 1./4 * e("imae,kjlcbd,le,ikjacb->md", eris.oovv, a3, t1, t3)  # d163_ov
    He += 1./12 * e("jkae,limdcb,ia,jkldcb->me", eris.oovv, a3, t1, t3)  # d164_ov
    He += 1./4 * e("jkba,miledc,ia,jkmbed->lc", eris.oovv, a3, t1, t3)  # d165_ov
    HHHeee -= 1./2 * p("...aab", e("imcb,kljade,ia,me->kljcbd", eris.oovv, a3, t1, t1))  # d166_ooovvv
    He += 1./4 * e("jiab,klmced,jd,ie,klac->mb", eris.oovv, a3, t1, t1, t2)  # d167_ov
    He -= 1./8 * e("ijba,lmkcde,ic,je,lmba->kd", eris.oovv, a3, t1, t1, t2)  # d168_ov
    HHHeee += p("...abb", e("ijab,lmkced,jc,ib->lmkaed", eris.oovv, a3, t1, t1))  # d169_ooovvv
    He += 1./2 * e("jiab,klmcde,ia,je,klbc->md", eris.oovv, a3, t1, t1, t2)  # d170_ov
    HHee -= 1./2 * p("..ab", p("ab..", e("lmed,ijkacb,mb,ijae->lkdc", eris.oovv, a3, t1, t2)))  # d171_oovv
    HHee += 1./4 * p("ab..", e("imed,kljacb,ia,kled->mjcb", eris.oovv, a3, t1, t2))  # d172_oovv
    HHee += 1./2 * e("kied,jlmcba,ia,jkcb->lmed", eris.oovv, a3, t1, t2)  # d173_oovv
    HHee -= p("..ab", e("lmed,ikjbac,ma,ilbe->kjdc", eris.oovv, a3, t1, t2))  # d174_oovv
    HHee += 1./2 * e("jkba,ilmced,kc,ijba->lmed", eris.oovv, a3, t1, t2)  # d175_oovv
    He -= 1./12 * e("lmae,jkicbd,ld,jkiacb->me", eris.oovv, a3, t1, t3)  # d176_ov
    He += 1./12 * e("imed,kljbac,ia,kljedb->mc", eris.oovv, a3, t1, t3)  # d177_ov
    He += 1./4 * e("ijba,mlkdce,ie,jmlbdc->ka", eris.oovv, a3, t1, t3)  # d178_ov
    He -= 1./4 * e("ijba,lmkdce,je,ilmbad->kc", eris.oovv, a3, t1, t3)  # d179_ov
    HHee -= 1./2 * p("ab..", e("klac,ijmbed,kc,ijab->lmed", eris.oovv, a3, t1, t2))  # d180_oovv
    HHee += 1./2 * p("..ab", e("imce,jklbad,me,ijba->klcd", eris.oovv, a3, t1, t2))  # d181_oovv
    HHee += e("jkbc,ilmaed,kc,ijab->lmed", eris.oovv, a3, t1, t2)  # d182_oovv
    He -= 1./12 * e("jiab,lmkedc,ib,lmkaed->jc", eris.oovv, a3, t1, t3)  # d183_ov
    He += 1./12 * e("kjba,lmiedc,jb,klmedc->ia", eris.oovv, a3, t1, t3)  # d184_ov
    He += 1./4 * e("ijab,lmkedc,jb,ilmaed->kc", eris.oovv, a3, t1, t3)  # d185_ov
    HHHeee -= 1./2 * p("...abb", p("aab...", e("lmed,ijkacb,ijea->lmkdcb", eris.oovv, a3, t2)))  # d186_ooovvv
    He += 1./4 * e("lmed,kjicba,jida,mkcb->le", eris.oovv, a3, t2, t2)  # d187_ov
    He -= 1./2 * e("mlae,kijdbc,ijab,mked->lc", eris.oovv, a3, t2, t2)  # d188_ov
    He -= 1./8 * e("ijab,klmedc,klac,ijed->mb", eris.oovv, a3, t2, t2)  # d189_ov
    He -= 1./4 * e("jkbd,lmicea,jkbc,lmde->ia", eris.oovv, a3, t2, t2)  # d190_ov
    HHHeee += 1./4 * p("aab...", e("ijba,kmledc,mlba->ijkedc", eris.oovv, a3, t2))  # d191_ooovvv
    He -= 1./8 * e("mlba,kijdce,ijba,mkdc->le", eris.oovv, a3, t2, t2)  # d192_ov
    He += 1./16 * e("lmed,ijkbac,ijed,lmba->kc", eris.oovv, a3, t2, t2)  # d193_ov
    HHHeee -= 1./2 * p("...aab", p("abb...", e("ikdc,jmlbae,ijba->kmldce", eris.oovv, a3, t2)))  # d194_ooovvv
    He += 1./2 * e("ijba,mlkedc,jldc,imbe->ka", eris.oovv, a3, t2, t2)  # d195_ov
    He -= 1./4 * e("jiba,mlkedc,iled,jmba->kc", eris.oovv, a3, t2, t2)  # d196_ov
    HHHeee += p("...abb", p("abb...", e("ijab,mlkedc,imae->jlkbdc", eris.oovv, a3, t2)))  # d197_ooovvv
    He += 1./2 * e("lmde,ikjacb,lkdc,miea->jb", eris.oovv, a3, t2, t2)  # d198_ov
    HHHeee -= 1./2 * p("abb...", e("ijba,mlkedc,imba->jlkedc", eris.oovv, a3, t2))  # d199_ooovvv
    HHHeee += 1./4 * p("...aab", e("ijba,lmkdce,ijdc->lmkbae", eris.oovv, a3, t2))  # d200_ooovvv
    HHHeee += 1./2 * p("...abb", e("ijab,lmkdce,ijbe->lmkadc", eris.oovv, a3, t2))  # d201_ooovvv
    HHee += 1./12 * p("..ab", e("ijca,lmkedb,lmkced->ijab", eris.oovv, a3, t3))  # d202_oovv
    HHee += 1./12 * e("lmed,jkiacb,jkieda->lmcb", eris.oovv, a3, t3)  # d203_oovv
    HHee += 1./12 * p("ab..", e("ijba,klmedc,ikledc->jmba", eris.oovv, a3, t3))  # d204_oovv
    HHee += 1./4 * p("..ab", p("ab..", e("ijab,lmkedc,ilmaed->jkbc", eris.oovv, a3, t3)))  # d205_oovv
    HHee += 1./4 * p("ab..", e("kmcb,jilaed,kjicba->mled", eris.oovv, a3, t3))  # d206_oovv
    HHee += 1./12 * e("jked,ilmcba,jkicba->lmed", eris.oovv, a3, t3)  # d207_oovv
    HHee -= 1./4 * p("..ab", e("klac,ijmbed,klmced->ijab", eris.oovv, a3, t3))  # d208_oovv
    HHee += 1./4 * e("lmed,ikjacb,lmieda->kjcb", eris.oovv, a3, t3)  # d209_oovv
    HHee -= p("..ab", e("jkai,ib->jkab", eris.oovo, a1))  # d210_oovv
    He += e("ikaj,jb,ib->ka", eris.oovo, a1, t1)  # d211_ov
    He += e("jiak,kb,ia->jb", eris.oovo, a1, t1)  # d212_ov
    HHHeee -= p("...abb", p("aab...", e("jkai,ilcb->jklacb", eris.oovo, a2)))  # d213_ooovvv
    HHee -= e("klci,ijba,jc->klba", eris.oovo, a2, t1)  # d214_oovv
    He += e("jkal,licb,ia,kc->jb", eris.oovo, a2, t1, t1)  # d215_ov
    HHee -= p("..ab", p("ab..", e("kjai,ilbc,jb->klac", eris.oovo, a2, t1)))  # d216_oovv
    He += 1./2 * e("lick,kjab,ia,lb->jc", eris.oovo, a2, t1, t1)  # d217_ov
    He += e("kjai,ilbc,kb,ja->lc", eris.oovo, a2, t1, t1)  # d218_ov
    HHee -= p("ab..", e("kicl,ljba,kc->ijba", eris.oovo, a2, t1))  # d219_oovv
    He += 1./2 * e("jkai,ilcb,jlcb->ka", eris.oovo, a2, t2)  # d220_ov
    He -= e("klcj,jiab,lica->kb", eris.oovo, a2, t2)  # d221_ov
    He -= 1./4 * e("jkai,ilcb,jkcb->la", eris.oovo, a2, t2)  # d222_ov
    He -= 1./2 * e("klbj,jica,klbc->ia", eris.oovo, a2, t2)  # d223_ov
    HHHeee += p("aab...", e("ijdl,lkmcba,md->ijkcba", eris.oovo, a3, t1))  # d224_ooovvv
    HHee += p("ab..", e("mldj,jkicba,id,mc->lkba", eris.oovo, a3, t1, t1))  # d225_oovv
    He += 1./2 * e("ijak,kmlcbd,ic,ma,jd->lb", eris.oovo, a3, t1, t1, t1)  # d226_ov
    He -= 1./2 * e("jmal,lkicbd,ia,jkcb->md", eris.oovo, a3, t1, t2)  # d227_ov
    He += 1./4 * e("klbj,jimdca,mb,kldc->ia", eris.oovo, a3, t1, t2)  # d228_ov
    HHHeee += p("...abb", p("abb...", e("ildm,mkjacb,ia->lkjdcb", eris.oovo, a3, t1)))  # d229_ooovvv
    HHee += 1./2 * p("..ab", e("imbl,ljkdac,ia,md->jkbc", eris.oovo, a3, t1, t1))  # d230_oovv
    HHee += e("imdl,ljkacb,ia,md->jkcb", eris.oovo, a3, t1, t1)  # d231_oovv
    He += 1./2 * e("lmdi,ikjacb,ma,kjdb->lc", eris.oovo, a3, t1, t2)  # d232_ov
    He += 1./2 * e("kiaj,jlmdcb,ib,kmdc->la", eris.oovo, a3, t1, t2)  # d233_ov
    He -= e("ikbj,jmldac,ia,klbc->md", eris.oovo, a3, t1, t2)  # d234_ov
    HHHeee += p("abb...", e("ildm,mkjcba,ld->ikjcba", eris.oovo, a3, t1))  # d235_ooovvv
    He += 1./2 * e("lmdi,ijkbac,md,ljba->kc", eris.oovo, a3, t1, t2)  # d236_ov
    HHee -= 1./2 * e("lmdi,ikjbac,kjdc->lmba", eris.oovo, a3, t2)  # d237_oovv
    HHee -= 1./2 * p("..ab", p("ab..", e("jiak,kmldcb,imdc->jlab", eris.oovo, a3, t2)))  # d238_oovv
    HHee -= p("ab..", e("mldi,ikjcba,mjda->lkcb", eris.oovo, a3, t2))  # d239_oovv
    HHee -= 1./4 * p("..ab", e("klam,mijbdc,kldc->ijab", eris.oovo, a3, t2))  # d240_oovv
    HHee -= 1./2 * e("kldm,mijbac,kldc->ijba", eris.oovo, a3, t2)  # d241_oovv
    He -= 1./12 * e("ijak,kmldcb,jmldcb->ia", eris.oovo, a3, t3)  # d242_ov
    He -= 1./4 * e("mldk,kjicba,mjidba->lc", eris.oovo, a3, t3)  # d243_ov
    He -= 1./12 * e("jkam,mildcb,jkldcb->ia", eris.oovo, a3, t3)  # d244_ov
    He -= 1./4 * e("lmdj,jikbac,lmidba->kc", eris.oovo, a3, t3)  # d245_ov
    HHee += 1./2 * e("ijkl,klba->ijba", eris.oooo, a2)  # d246_oovv
    He -= 1./2 * e("lijk,jkab,lb->ia", eris.oooo, a2, t1)  # d247_ov
    HHHeee += 1./2 * p("aab...", e("lmij,ijkcba->lmkcba", eris.oooo, a3))  # d248_ooovvv
    HHee += 1./2 * p("ab..", e("mikl,kljcba,mc->ijba", eris.oooo, a3, t1))  # d249_oovv
    He += 1./4 * e("imkl,kljacb,ia,mc->jb", eris.oooo, a3, t1, t1)  # d250_ov
    He -= 1./4 * e("iklm,jlmbac,ijba->kc", eris.oooo, a3, t2)  # d251_ov
    He += 1./8 * e("lmjk,jkicba,lmcb->ia", eris.oooo, a3, t2)  # d252_ov
    return He, HHee, HHHeee

if __name__ == '__main__':
    
    import numpy as np
    from pyscf import gto, scf, cc
    import Eris
    import utilities

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'sto3g'
    mol.spin = 0
    mol.build()

    # generalize HF and CC
    mgf = scf.GHF(mol)
    mgf.kernel()

    mycc = cc.GCCSD(mgf)
    eris = Eris.geris(mycc)
    nocc = mycc.nocc
    nvir = mgf.mo_coeff.shape[0]-nocc

    print()
    print('################')
    print('# R/L equations ')
    print('################')
    print()
    
    ts = np.random.random((nocc//2,nvir//2))*0.1
    ls = np.random.random((nocc//2,nvir//2))*0.1
    rs = np.random.random((nocc//2,nvir//2))*0.1
    ts = utilities.convert_r_to_g_amp(ts)
    ls = utilities.convert_r_to_g_amp(ls)
    rs = utilities.convert_r_to_g_amp(rs)
    r0 = 0.1
    l0 = 0.1

    print('R0= ', R10eq(ts,rs,r0,eris))
    print('L0= ', es_L10eq(ts, ls, l0, eris))
    print('R equations=')
    print(R1eq(ts,rs,r0,eris))
    print('L equations=')
    print(es_L1eq(ts, ls, l0, eris))
    

