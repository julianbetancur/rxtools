import numpy as np
import vtk
from vtk.util import numpy_support

def get_metal_volume( vol_spacing, xyz_bounds, polydata_envelop, list_polydata_cores, list_transform_cores, fill_in_value=1.0):
    """
    Create a voxelized image of the metal volume defined by the STL surface mesh.
    xyz_bounds: [xmin, xmax, ymin, ymax, zmin, zmax]
    """

    nx = int(np.ceil((xyz_bounds[1] - xyz_bounds[0]) / vol_spacing[0]))
    ny = int(np.ceil((xyz_bounds[3] - xyz_bounds[2]) / vol_spacing[1]))
    nz = int(np.ceil((xyz_bounds[5] - xyz_bounds[4]) / vol_spacing[2]))

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise RuntimeError("Degenerate bounds; check STL units or geometry.")

    vol_size = (nx + 1, ny + 1, nz + 1)
    # create reference volume
    img = vtk.vtkImageData()
    img.SetOrigin(xyz_bounds[0], xyz_bounds[2], xyz_bounds[4])
    img.SetSpacing(vol_spacing[0], vol_spacing[1], vol_spacing[2])
    img.SetDimensions(*vol_size)
    img.AllocateScalars(vtk.VTK_FLOAT, 1)
    
    # Fill with mu; we will zero outside using the stencil
    img_np = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
    img_np.fill(fill_in_value)

    if len(list_polydata_cores) != len(list_transform_cores):
        raise ValueError("Length of list_polydata_cores and list_transform_cores must be the same.")
        return img_np

    #poyldata stencil object
    poly2stencil = vtk.vtkPolyDataToImageStencil()
    poly2stencil.SetInputData(polydata_envelop)
    poly2stencil.SetOutputOrigin(img.GetOrigin())
    poly2stencil.SetOutputSpacing(img.GetSpacing())
    extent = (0, vol_size[0]-1, 0, vol_size[1]-1, 0, vol_size[2]-1)
    poly2stencil.SetOutputWholeExtent(*extent)
    poly2stencil.Update()

    stenciler = vtk.vtkImageStencil()
    stenciler.SetInputData(img)
    stenciler.SetStencilConnection(poly2stencil.GetOutputPort())
    stenciler.SetBackgroundValue(0.0)  # everything *outside* STL -> 0 attenuation
    stenciler.ReverseStencilOff()
    stenciler.Update()

    voxel_img = stenciler.GetOutput()

    # ========= 4) VTK image -> NumPy volume (z, y, x) =========
    vol_scalars = voxel_img.GetPointData().GetScalars()
    vol_np_flat = numpy_support.vtk_to_numpy(vol_scalars).astype(np.float32)

    # reshape point grid then drop last index in each dim to get cell-centered voxels
    nzp, nyp, nxp = vol_size[2], vol_size[1], vol_size[0]
    vol_points = vol_np_flat.reshape((nzp, nyp, nxp))  # (z, y, x) order
    volume = vol_points[:nz, :ny, :nx]                 # (z, y, x), float32

    return volume

    