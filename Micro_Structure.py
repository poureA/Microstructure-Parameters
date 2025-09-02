# Import necessaries
import numpy as np
import tifffile
import os
import pumapy as puma
from sailfish.geo import LBGeometry2D
from sailfish.lb_single import LBFluidSim
from sailfish.controller import LBSimulationController


class Micro_Structure:
    '''
    A class to compute microstructural and transport properties 
    (surface area, mean intercept length, permeability, tortuosity, etc.)
    from 3D image-based data using PuMA and Sailfish.

    Attributes
    ----------
    img : numpy.ndarray
        The 3D image stack (binary or grayscale volume).
    ws : puma.Workspace
        PuMA workspace object containing the 3D microstructure.
    '''
    def __init__(self, image, ws):
        '''
        Initialize the Micro_Structure class.

        Parameters
        ----------
        image : numpy.ndarray
            3D array representing the microstructure (e.g., TIFF stack).
        ws : puma.Workspace
            PuMA workspace object for the microstructure.
        '''
        self.img = image
        self.ws = ws

    def Specific_Surface_Area(self):
        '''
        Compute the surface area and specific surface area.

        Returns
        -------
        Surface_Area : float
            Total surface area of the solid phase.
        Specific_SurfaceArea : float
            Surface area normalized by volume (specific surface area).
        '''
        Surface_Area, Specific_SurfaceArea = puma.surface_area(
            self.ws, (0, 0), (255, 255)
        )
        return Surface_Area, Specific_SurfaceArea

    def Mean_Intercept_Length(self):
        '''
        Compute the mean intercept length of the porous structure.

        Returns
        -------
        lean_lntercept_length : float
            Mean intercept length of the structure.
        '''
        lean_lntercept_length = puma.mean_intercept_length(
            self.ws, cutoff=(0, 0)
        )
        return lean_lntercept_length

    def Permeability(self):
        '''
        Compute permeability in x, y, and z directions.

        Returns
        -------
        permeability_X : float
            Permeability along x-direction.
        permeability_Y : float
            Permeability along y-direction.
        permeability_Z : float
            Permeability along z-direction.
        '''
        permeability_X = puma.compute_permeability(
            self.ws, cutoff=(0, 0), direction='x',
            tol=1e-4, maxiter=10000, solver='bicgstab'
        )
        permeability_Y = puma.compute_permeability(
            self.ws, cutoff=(0, 0), direction='Y',
            tol=1e-4, maxiter=10000, solver='bicgstab'
        )
        permeability_Z = puma.compute_permeability(
            self.ws, cutoff=(0, 0), direction='Z',
            tol=1e-4, maxiter=10000, solver='bicgstab'
        )
        return permeability_X['k'], permeability_Y['k'], permeability_Z['k']

    def Tortuosity(self):
        '''
        Compute tortuosity in x, y, and z directions.

        Returns
        -------
        tortuosity_X : float
            Tortuosity along x-direction.
        tortuosity_Y : float
            Tortuosity along y-direction.
        tortuosity_Z : float
            Tortuosity along z-direction.
        '''
        tortuosity_X = puma.compute_tortuosity(
            self.ws, cutoff=(0, 0), direction='x',
            maxiter=10000, tol=1e-5, solver='cg'
        )
        tortuosity_Y = puma.compute_tortuosity(
            self.ws, cutoff=(0, 0), direction='Y',
            maxiter=10000, tol=1e-5, solver='cg'
        )
        tortuosity_Z = puma.compute_tortuosity(
            self.ws, cutoff=(0, 0), direction='Z',
            maxiter=10000, tol=1e-5, solver='cg'
        )
        return tortuosity_X['tau'], tortuosity_Y['tau'], tortuosity_Z['tau']

    def kozeny_carman_surface(self, porosity, C=5):
        '''
        Calculate permeability using the Kozeny–Carman equation (surface area form).

        Parameters
        ----------
        porosity : float
            Porosity (0 < phi < 1). 
            Note: If porosity is given in %, divide by 100 before calling.
        C : float, optional
            Kozeny constant (default=5, typically 4–6 for packed spheres).

        Returns
        -------
        k : float
            Estimated permeability [m^2].

        Raises
        ------
        ValueError
            If porosity is not between 0 and 1.
            If specific surface area is not positive.
        '''
        if porosity <= 0 or porosity >= 1:
            raise ValueError("Porosity must be between 0 and 1 (fraction).")
        if self.Specific_Surface_Area() <= 0:
            raise ValueError("Specific surface area must be positive.")

        k = (1 / C) * (porosity**3) / (
            (1 - porosity)**2 * self.Specific_Surface_Area()**2
        )
        return k

    def run_lbm(self, inlet_vel=0.05):
        '''
        Run a 2D Lattice Boltzmann Method (LBM) simulation on a slice 
        of the 3D image using Sailfish.

        Parameters
        ----------
        inlet_vel : float, optional
            Inlet velocity for the simulation (default=0.05).

        Raises
        ------
        ValueError
            If self.img is None.

        Returns
        -------
        None
            Runs the Sailfish solver (results are handled by Sailfish).
        '''
        if self.img is None:
            raise ValueError("Mask not defined. Run get_ice_part() first.")
        
        # Use the middle slice of the volume for 2D LBM
        geom_mask = self.img[self.img.shape[0]//2]

        # Define geometry for Sailfish
        class IceGeometry(LBGeometry2D):
            def __init__(self, shape, mask):
                super(IceGeometry, self).__init__(shape)
                self._mask = mask

            def define_nodes(self):
                # Ice = solid walls
                self.set_geo(self._mask > 0, self.NODE_WALL)

        # Define LBM simulation
        class IceSim(LBFluidSim):
            @classmethod
            def add_options(cls, group):
                group.add_argument('--inlet-vel', type=float, default=inlet_vel)

            def boundary_conditions(self, hx, hy):
                # Inlet condition
                self.velocity_bc(hx == 0, (self.config.inlet_vel, 0.0))
                # Outlet condition
                self.outlet_bc(hx == self.shape[0]-1)

            def geometry_class(self):
                return lambda shape: IceGeometry(shape, geom_mask)

        # Run Sailfish simulation
        LBSimulationController(IceSim, LBGeometry2D).run()


def Create_a_sample():
    '''
    Create a synthetic 3D binary volume for testing.

    Creates a volume of size (100, 200, 200) with a spherical 
    "ice blob" added in the center slices. Saves it as a TIFF stack.

    Returns
    -------
    None
    '''
    # Create a synthetic 3D binary volume (100 slices of 200x200 pixels)
    data = np.zeros((100, 200, 200), dtype=np.uint8)
    # Add a few "ice blobs"
    rr, cc = np.ogrid[:200, :200]
    mask = (rr-100)**2 + (cc-100)**2 < 40**2
    for i in range(30,70):
        data[i][mask] = 255  # white = ice
    tifffile.imwrite("test_stack.tif", data)


if __name__ == '__main__':
    # Create a sample in order to test the following functions
    if 'test_stack.tif' not in os.listdir(os.getcwd()):
        Create_a_sample()
        sample = tifffile.imread('test_stack.tif')
    else:
        sample = tifffile.imread('test_stack.tif')

    # Convert into puma workspace
    ws = puma.Workspace.from_array(sample, voxel_length=1.0)

    print('Result for test_stack.tif:')
    test_case = Micro_Structure(sample, ws)
    print('Specific Surface Area :', test_case.Specific_Surface_Area())
    print('Mean Intercept Length :', test_case.Mean_Intercept_Length())
    print('Permeability (x,y,z) :', test_case.Permeability())
    print('Tortuosity (x,y,z) :', test_case.Tortuosity())
    print('kozeny_carman_surface :', test_case.kozeny_carman_surface(0.5))
    print('LBM :', test_case.run_lbm())