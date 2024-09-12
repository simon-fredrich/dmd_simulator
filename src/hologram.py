from metadata import MetaData
import numpy as np
import matplotlib.pyplot as plt
from aotools.functions import phaseFromZernikes
import scipy.special as sp

class Hologram():
    def __init__(self, meta:MetaData) -> None:
        self.nr_m=meta.nr_m
        self.wavelength=meta.wavelength
        self.phase: float
        self.amplitude: float
        self.X, self.Y=np.ogrid[0:self.nr_m:self.nr_m*1j,\
                                0:self.nr_m:self.nr_m*1j]
        self.H: np.ndarray
        self.complex_pattern: np.ndarray
        self.zernike_coefficients: np.ndarray

    def get_disk_mask(self, shape, radius, center = None):
        '''
        Generate a binary mask with value 1 inside a disk, 0 elsewhere
        :param shape: list of integer, shape of the returned array
        :radius: integer, radius of the disk
        :center: list of integers, position of the center
        :return: numpy array, the resulting binary mask
        '''
        if not center:
            center = (shape[0]//2,shape[1]//2)
        X,Y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
        mask = (Y-center[0])**2+(X-center[1])**2 < radius**2
        return mask.astype(int)

    def complex_mask_from_zernike_coeff(self, shape, radius, center, vec):
        '''
        Generate a complex phase mask from a vector containting the coefficient of the first Zernike polynoms.
        :param DMD_resolution: list of integers, contains the resolution of the DMD, e.g. [1920,1200]
        :param: integer, radius of the illumination disk on the DMD
        :center: list of integers, contains the position of the center of the illumination disk
        :center: list of float, the coefficient of the first Zernike polynoms
        '''
        self.zernike_coefficients=vec
        # Generate a complex phase mask from the coefficients
        zern_mask = np.exp(1j*phaseFromZernikes(vec,2*radius))
        zern_mask/=np.max(np.abs(zern_mask))
        print(np.max(np.abs(zern_mask)))
        # We want the amplitude to be 0 outside the disk, we fist generate a binary disk mask
        amp_mask = self.get_disk_mask([2*radius]*2,radius)
        print(np.max(np.abs(amp_mask)))
        # put the Zernik mask at the right position and multiply by the disk mask
        mask = np.zeros(shape = shape, dtype=complex)
        mask[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius] = zern_mask*amp_mask
        return mask

    def get_lee_holo(self, period, vec=[0., 0., 5., 5.], angle=0, nbits = 8):
        '''
        complex_pattern: the input amplitude and phase pattern we want to generate.
                        its amplitude should be <= 1
        period: period of the grating of the Lee hologram pattern.
        res: resolution of your pattern. Tuple of integers.
        amgle: angle of the Lee hologram grating.
        nbits: number of bits of the image. nbits = 8 will return pattern with values 0 and 255.
        
        The phase is encoded in the displacement of fringes.
        The amplitude is encoded by removing lines orthogonally to the Lee grating.
        '''
        res=[self.nr_m]*2
        complex_pattern=self.complex_mask_from_zernike_coeff(res, self.nr_m//5, [self.nr_m//2]*2, vec)
        self.complex_pattern=complex_pattern
        angle*=np.pi/180
        assert not np.max(np.abs(complex_pattern)) > 1.
        omega = 2.*np.pi/period
        dmd_res = complex_pattern.shape
        X,Y = np.meshgrid(np.arange(res[1]),np.arange(res[0]))
        tilt_angle = (np.cos(angle)*X+np.sin(angle)*Y)
        amplitude_term = (np.abs(complex_pattern) + np.arcsin(np.cos(tilt_angle*omega))/np.pi)
        amplitude_term = amplitude_term > 0.5
        phase_term = (1.+np.cos((np.cos(angle)*X+np.sin(angle)*Y)*omega+np.angle(complex_pattern)))/2
        phase_term = phase_term > 0.5
        mask = amplitude_term * phase_term
            
        mask = (2**nbits-1)*mask
        mask = mask.astype(np.uint8)
        return np.where(mask>0, 1, 0)

    def create(self, lambda_x, lambda_y, phase, amplitude):
        """
        Create a binary hologram with the given parameters.

        Parameters
        ----------
        lambda_x, lambda_y : float
            Spatial frequencies in the x and y directions.
        phase : float
            Phase of the hologram.
        amplitude : float
            Amplitude of the hologram.

        Returns
        -------
        H : numpy array
            The hologram pattern.
        """
        self.phase=phase
        self.amplitude=amplitude
        self.lambda_x=lambda_x
        self.lambda_y=lambda_y
        H=1/2+1/2*np.sign(
            np.cos(2*np.pi*(self.X/lambda_x+self.Y/lambda_y)+phase)-
            np.cos(np.arcsin(amplitude)))
        
        self.H=H
        return H
    
    def display(self):
        plt.imshow(self.H, cmap=plt.cm.gray)
        plt.show()
        

    def pattern_on(self):
        return np.ones((self.nr_m, self.nr_m))
    
    def pattern_off(self):
        return np.zeros((self.nr_m, self.nr_m))
    
    def grating(self, period):
        return 1/2+1/2*np.sign(np.cos(2*np.pi/period*(self.X+self.Y)))
    

def get_disk_mask(shape, radius, center = None):
    '''
    Generate a binary mask with value 1 inside a disk, 0 elsewhere
    :param shape: list of integer, shape of the returned array
    :radius: integer, radius of the disk
    :center: list of integers, position of the center
    :return: numpy array, the resulting binary mask
    '''
    if not center:
        center = (shape[0]//2,shape[1]//2)
    X,Y = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))
    mask = (Y-center[0])**2+(X-center[1])**2 < radius**2
    return mask.astype(int)

def complex_mask_from_zernike_coeff(shape, radius, center, vec):
    '''
    Generate a complex phase mask from a vector containting the coefficient of the first Zernike polynoms.
    :param DMD_resolution: list of integers, contains the resolution of the DMD, e.g. [1920,1200]
    :param: integer, radius of the illumination disk on the DMD
    :center: list of integers, contains the position of the center of the illumination disk
    :center: list of float, the coefficient of the first Zernike polynoms
    '''
    # Generate a complex phase mask from the coefficients
    zern_mask = np.exp(1j*phaseFromZernikes(vec,2*radius))
    zern_mask/=np.max(np.abs(zern_mask))
    # We want the amplitude to be 0 outside the disk, we fist generate a binary disk mask
    amp_mask = get_disk_mask([2*radius]*2,radius)
    # put the Zernik mask at the right position and multiply by the disk mask
    mask = np.zeros(shape = shape, dtype=complex)
    mask[center[0]-radius:center[0]+radius, center[1]-radius:center[1]+radius] = zern_mask*amp_mask
    return mask

def get_lee_holo(complex_pattern, res, period, angle=0, nbits = 8):
    '''
    complex_pattern: the input amplitude and phase pattern we want to generate.
                     its amplitude should be <= 1
    period: period of the grating of the Lee hologram pattern.
    res: resolution of your pattern. Tuple of integers.
    amgle: angle of the Lee hologram grating.
    nbits: number of bits of the image. nbits = 8 will return pattern with values 0 and 255.
    
    The phase is encoded in the displacement of fringes.
    The amplitude is encoded by removing lines orthogonally to the Lee grating.
    '''
    angle*=np.pi/180
    assert not np.max(np.abs(complex_pattern)) > 1.
    omega = 2.*np.pi/period
    dmd_res = complex_pattern.shape
    X,Y = np.meshgrid(np.arange(res[1]),np.arange(res[0]))
    tilt_angle = (np.cos(angle)*X+np.sin(angle)*Y)
    amplitude_term = (np.abs(complex_pattern) + np.arcsin(np.cos(tilt_angle*omega))/np.pi)
    amplitude_term = amplitude_term > 0.5
    phase_term = (1.+np.cos((np.cos(angle)*X+np.sin(angle)*Y)*omega+np.angle(complex_pattern)))/2
    phase_term = phase_term > 0.5
    mask = amplitude_term * phase_term
          
    mask = (2**nbits-1)*mask
    return mask.astype(np.uint8)


def radial_polynomial(n, m, rho):
    """
    Compute the radial component of the Zernike polynomial.
    """
    R = np.zeros_like(rho)
    for k in range((n - abs(m)) // 2 + 1):
        R += ((-1)**k * sp.factorial(n - k) /\
        (sp.factorial(k) * sp.factorial((n + abs(m)) // 2 - k) *\
        sp.factorial((n - abs(m)) // 2 - k))) * rho**(n - 2*k)
    return R

def zernike_polynomial(n, m, rho, phi):
    """
    Compute the Zernike polynomial.
    """
    if m >= 0:
        return radial_polynomial(n, m, rho) * np.cos(m * phi)
    else:
        return radial_polynomial(n, -m, rho) * np.sin(-m * phi)