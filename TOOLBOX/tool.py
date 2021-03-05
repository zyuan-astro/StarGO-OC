import numpy as np
from math import *
from astropy.coordinates import SkyCoord
import astropy.units as units
from scipy.stats import binom
import scipy.stats
from astroML.density_estimation import EmpiricalDistribution


# in LSR frame
U_sun, V_sun, W_sun = 11.1, 12.24, 7.25
X_sun = 8.3
V_LSR = 232

def cal_exptime(mag, sn=60):
    
    exptime = ((sn/100)**2.)*900*np.exp(0.4*(mag-12))

    return exptime
    

def RE_GENERATE_SAMPLE(orig_sample_data, sample_size = 10000):
    
    dim = orig_sample_data.shape[1]
    new_sample_data = np.empty([sample_size,dim])
    for i in range(dim):
        
        new_sample_data[:,i] = EmpiricalDistribution(orig_sample_data[:,i]).rvs(sample_size)

    
    return new_sample_data

    
    
    
    
def prob(n, p, n_tot):
    
    total_p = 0
    for k in range(0, n):  # DO NOT FORGET THAT THE LAST INDEX IS NOT USED
    
        total_p += np.float64(binom.pmf(k, n_tot, p))
    return total_p

def d_ga(lon1, lat1, lon2, lat2): # -90 deg < la < 90 deg
    
    

    c1 = SkyCoord(lon1*units.radian, lat1*units.radian)
    c2 = SkyCoord(lon2*units.radian, lat2*units.radian)
    sep = c1.separation(c2).radian
    
    return sep


def re_cal_u_angle(w):
    
    
    
    lon = w[:,:,3] * 2 * pi
    lat = w[:,:,2] * pi - pi /2.
    
    dw_row = np.diff(w, axis = 0)
    dw_col = np.diff(w, axis = 1)
        
    dw_ga_row = d_ga(lon[:-1,:] , lat[:-1,:], lon[1:,:], lat[1:,:])
    dw_ga_col = d_ga(lon[:,:-1] , lat[:,:-1], lon[:,1:], lat[:,1:])


    u_row = np.sqrt((dw_row[:,:,:2]**2.0).sum(-1)+(dw_ga_row/pi)**2.)
    u_col = np.sqrt((dw_col[:,:,:2]**2.0).sum(-1)+(dw_ga_col/pi)**2.)

    u = u_row[:, :-1] + u_col[:-1, :]

    return u


def re_cal_vw_angle(v,w):
    
    d_vw = v - w

    lon_v = v[:,3] * 2 * pi
    lat_v = v[:,2] * pi - pi /2.
    
    lon_w = w[:,3] * 2 * pi
    lat_w = w[:,2] * pi - pi /2.
    
    d_ga_vw = d_ga(lon_v, lat_v, lon_w, lat_w)

    u_vw= np.sqrt((d_vw[:,:2]**2.0).sum(-1)+(d_ga_vw/pi)**2.)

    
    return u_vw



def shift(ra):
        
    ra1 = np.copy(ra)

    ind = ra1 > 180

    ra1[ind] =  ra1[ind] - 360
    
    
    return ra1

#v_gsr = rv + U_sun * np.cos(gl*pi/180)*np.cos(gb*pi/180) + V_LSR*np.sin(gl*pi/180)*np.cos(gb*pi/180) + W_sun*np.sin(gb*pi/180)

def cal_v_gsr(gl,gb,vlos):

    v_gsr = vlos + U_sun * np.cos(gl*pi/180)*np.cos(gb*pi/180) + V_LSR*np.sin(gl*pi/180)*np.cos(gb*pi/180) + W_sun*np.sin(gb*pi/180)
    
    return v_gsr





def radec2sag(ra,dec,Degree=True):

    if Degree:
        ra_tmp = ra*pi/180
        dec_tmp = dec*pi/180
    else:
        ra_tmp = ra
        dec_tmp = dec
        
    a11, a12, a13 = -0.93595354, -0.31910658,  0.14886895
    a21, a22, a23 =  0.21215555, -0.84846291, -0.48487186
    b11, b12, b13 =  0.28103559, -0.42223415,  0.86182209
    sag_lam = np.arctan2(a11*np.cos(ra_tmp)*np.cos(dec_tmp) + 
                         a12*np.sin(ra_tmp)*np.cos(dec_tmp) + 
                         a13*np.sin(dec_tmp), 
                         a21*np.cos(ra_tmp)*np.cos(dec_tmp) + 
                         a22*np.sin(ra_tmp)*np.cos(dec_tmp) + 
                         a23*np.sin(dec_tmp))
    sag_bet = np.arcsin(b11*np.cos(ra_tmp)*np.cos(dec_tmp) + 
                        b12*np.sin(ra_tmp)*np.cos(dec_tmp) + 
                        b13*np.sin(dec_tmp))
    
    
    ind_lam = (sag_lam<0)
        
    sag_lam[ind_lam] = sag_lam[ind_lam] + 2*pi
        
    n = len(ra)
    sag_coo = np.zeros((n,2))
    if Degree:
        sag_coo[:,0] = sag_lam*180/pi
        sag_coo[:,1] = sag_bet*180/pi
    else:
        sag_coo[:,0] = sag_lam
        sag_coo[:,1] = sag_bet
    return sag_coo





def get_projection(t_i, p_i, l):
    
    t_i = t_i / 180 * pi
    p_i = p_i / 180 * pi
    
    r1 = (sin(t_i) * cos(p_i), sin(t_i) * sin(p_i), cos(t_i))
    r2 = (cos(t_i) * cos(p_i), cos(t_i) * sin(p_i), -sin(t_i))
    r3 = np.cross(r1, r2)
    
    
    l_m = np.sqrt((l**2.0).sum(-1))
    
    l1 = np.inner(l, r1)
    cos1 = l1 / l_m
    theta1 = np.arccos(cos1) * 180 / pi
    
    l2 = np.inner(l, r2)
    cos2 = l2 / l_m
    theta2 = np.arccos(cos2) * 180 / pi
    
    l3 = np.inner(l, r3)
    cos3 = l3 / l_m
    theta3 = np.arccos(cos3) * 180 / pi
    
    return theta1, l1, theta2, l2, theta3, l3

def get_angles_spherical(vect):
    
    x = vect[:, 0]
    y = vect[:, 1]
    z = vect[:, 2]
    m = np.sqrt((vect**2.0).sum(-1))
    m_xy = np.sqrt(x**2.0 + y**2.0)

    cos_t = z / m
    cos_p = x / m_xy
    sin_p = y / m_xy

    t = np.arccos(cos_t)
    p = np.arccos(cos_p)

    for i in range(len(p)):
    
        if (sin_p[i] < 0):
        
            p[i] = 2. * pi - p[i]
            
    t = t * 180 / pi
    p = p * 180 / pi
    
    
    
    return t, p


def get_cartisian_coord(r, b, l):
    
    b = b * pi / 180.
    l = l * pi / 180
    
    x = r * np.sin(b) * np.cos(l)
    
    y = r * np.sin(b) * np.sin(l)
    
    z = r * np.cos(b)
    
    return x, y, z



def get_angle(t1, p1, t2, p2):

    r1 = (sin(t1) * cos(p1), sin(t1) * sin(p1), cos(t1))
    r2 = (sin(t2) * cos(p2), sin(t2) * sin(p2), cos(t2))
    
    cos_theta = np.inner(r1, r2)
    theta = acos(cos_theta) * 180 / pi
    
    return theta
    
def get_direction(a):
    
    x = a[:, 0]
    y = a[:, 1]
    z = a[:, 2]
    
    a_m = np.sqrt((a**2.0).sum(-1))
    a_xy_m = np.sqrt(x**2.0 + y**2.0)
    
    cos_t = z / a_m
    cos_p = x / a_xy_m
    sin_p = y / a_xy_m
    
    t = np.arccos(cos_t)
    p = np.arccos(cos_p)

    for i in range(len(p)):
    
        if (sin_p[i] < 0):
        
            p[i] = 2. * pi - p[i]
    
    t = t * 180 / pi
    p = p * 180 / pi
    
    return t, p



def get_radial_projection(r, v):
    
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    
    r_m = np.sqrt((r**2.0).sum(-1))
    r_xy_m = np.sqrt(x**2.0 + y**2.0)
    
    cos_t = z / r_m
    sin_t = np.sqrt(1 - cos_t**2.)
    cos_p = x / r_xy_m
    sin_p = y / r_xy_m
    

    v_r = np.empty(len(x), dtype = np.float32)
    v_t = np.empty(len(x), dtype = np.float32)
    v_p = np.empty(len(x), dtype = np.float32)

    
    
    for i in range(len(x)):
        

        r0 = (sin_t[i] * cos_p[i], sin_t[i] * sin_p[i], cos_t[i])
        rt = (cos_t[i] * cos_p[i], cos_t[i] * sin_p[i], -sin_t[i])
        rp = np.cross(r0, rt)
        
        v_r[i] = np.inner(v[i, :], r0)
        v_t[i] = np.inner(v[i, :], rt)
        v_p[i] = np.inner(v[i, :], rp)

    
    
    return v_r, v_t, v_p

def get_cartisian_projection(r, v):
    
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    
    v_r = v[:, 0]
    v_t = v[:, 1]
    v_p = v[:, 2]
    
    r_m = np.sqrt((r**2.0).sum(-1))
    r_xy_m = np.sqrt(x**2.0 + y**2.0)
    
    cos_t = z / r_m
    sin_t = np.sqrt(1 - cos_t**2.)
    cos_p = x / r_xy_m
    sin_p = y / r_xy_m
  
  
    vx = np.empty(len(x))
    vy = np.empty(len(x))
    vz = np.empty(len(x))

    for i in range(len(x)):
        
        r0 = (sin_t[i] * cos_p[i], sin_t[i] * sin_p[i], cos_t[i])
        rt = (cos_t[i] * cos_p[i], cos_t[i] * sin_p[i], -sin_t[i])
        rp = np.cross(r0, rt)
        
        rx = (1, 0, 0)
        ry = (0, 1, 0)
        rz = (0, 0, 1)
        
        v = np.multiply(v_r[i], r0) + np.multiply(v_t[i], rt) + np.multiply(v_p[i], rp)
        vx[i] = np.inner(v, rx) 
        vy[i] = np.inner(v, ry) 
        vz[i] = np.inner(v, rz) 

    
    
    return vx, vy, vz


class mw_profile(object):
    
    def __init__(self, c, r_vir, m_vir):

        
        self.c = c
        self.r_vir = r_vir
        self.m_vir = m_vir
        
        self.rs = r_vir / c
        
        self.fc = log(1.0 + c) - c / (1.0 + c)
        
        
    def get_density(self, r):
        
        self.rhos = self.m_vir * 1e12 / (4. * pi * self.rs**3.0 * self.fc)
        x = r / self.rs
        rho = self.rhos / x / (1.0 + x)**2.0
        
        return rho
        
    def get_potential(self, r):
    
        
        
        phi = - self.m_vir * 1e12 * G / self.fc / r * np.log(1.0 + r / self.rs)
    
        return phi


    def get_velocity_circ(self, r):

        x = r / self.rs
    
        fx = np.log(1.0 + x) - x / (1 + x)
    
        m = self.m_vir / self.fc * fx
        v_c = np.sqrt(G * m * 1e12 / r)
    
    
        return v_c
    
    
def Cal_Sigma_6d(ra_e, dec_e, plx_e, pmra_e, pmdec_e, rv_e,
        ra_dec_cor, ra_plx_cor, ra_pmra_cor, ra_pmdec_cor,
        dec_plx_cor, dec_pmra_cor, dec_pmdec_cor, plx_pmra_cor,
        plx_pmdec_cor, pmra_pmdec_cor):
  
    # Build entries
    s00 = ra_e ** 2
    s11 = dec_e ** 2
    s22 = plx_e ** 2
    s33 = pmra_e ** 2
    s44 = pmdec_e ** 2
    s55 = rv_e ** 2

    s01 = ra_e * dec_e * ra_dec_cor
    s02 = ra_e * plx_e * ra_plx_cor
    s03 = ra_e * pmra_e * ra_pmra_cor
    s04 = ra_e * pmdec_e * ra_pmdec_cor
    s05 = 0

    s12 = dec_e * plx_e * dec_plx_cor
    s13 = dec_e * pmra_e * dec_pmra_cor
    s14 = dec_e * pmdec_e * dec_pmdec_cor
    s15 = 0

    s23 = plx_e * pmra_e * plx_pmra_cor
    s24 = plx_e * pmdec_e * plx_pmdec_cor
    s25 = 0

    s34 = pmra_e * pmdec_e * pmra_pmdec_cor
    s35 = 0

    s45 = 0
    
    
    sigma = np.array([
        [s00, s01, s02, s03, s04, s05],
        [s01, s11, s12, s13, s14, s15],
        [s02, s12, s22, s23, s24, s25],
        [s03, s13, s23, s33, s34, s35],
        [s04, s14, s24, s34, s44, s45],
        [s05, s15, s25, s35, s45, s55]
        ])
    
    
    return sigma

def Cal_Sigma_3d(plx_e, pmra_e, pmdec_e, plx_pmra_cor,plx_pmdec_cor, pmra_pmdec_cor):
  
    s00 = plx_e **2
    s11 = pmra_e ** 2
    s22 = pmdec_e ** 2
 



    s01 = plx_e * pmra_e * plx_pmra_cor
    s02 = plx_e * pmdec_e * plx_pmdec_cor

    s12 = pmra_e * pmdec_e * pmra_pmdec_cor

    sigma = np.array([
        [s00, s01, s02],
        [s01, s11, s12],
        [s02, s12, s22],
        ])
    return sigma



def Cal_Sigma_2d(pmra_e, pmdec_e, pmra_pmdec_cor):
  
    
    s11 = pmra_e ** 2
    s22 = pmdec_e ** 2

    s12 = pmra_e * pmdec_e * pmra_pmdec_cor

    sigma = np.array([
        [s11, s12],
        [s12, s22],
        ])
    return sigma