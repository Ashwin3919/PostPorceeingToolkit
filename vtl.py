import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import gc  # Importing the garbage collector
def read_vtk_to_numpy(file_path, scalar_name):
    """
    Read scalar data from a VTK file and convert it to a 3D NumPy array.

    Parameters:
        file_path (str): The path to the VTK file.
        scalar_name (str): The name of the scalar data to read.

    Returns:
        np.ndarray: The scalar data as a 3D NumPy array.
        vtk.vtkStructuredPoints: The VTK structured points data.
    """
    # Read the scalar data from the VTK file
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_path)
    reader.Update()
    data_vtk = reader.GetOutput()
    scalar_data_vtk = data_vtk.GetPointData().GetArray(scalar_name)
    
    if scalar_data_vtk is None:
        raise ValueError(f"The VTK file at {file_path} does not contain scalar data named {scalar_name}.")
    
    # Get the dimensions of the VTK data
    dimensions = data_vtk.GetDimensions()
    
    # Initialize a 3D NumPy array with the same dimensions
    scalar_data_np = np.zeros(dimensions)
    
    # Fill the NumPy array with the scalar data from the VTK array
    for k in range(dimensions[2]):
        for j in range(dimensions[1]):
            for i in range(dimensions[0]):
                scalar_data_np[i, j, k] = scalar_data_vtk.GetTuple1(k*dimensions[1]*dimensions[0] + j*dimensions[0] + i)
    
    return scalar_data_np, data_vtk



import vtk
import gc
from vtk.util.numpy_support import numpy_to_vtk

def write_numpy_to_vtk_3d(np_array, reference_grid, output_file_name, scalar_name="scalars"):
    """
    Write a 3D NumPy array to a VTK file as a scalar field, utilizing garbage collection.

    Parameters:
        np_array (np.ndarray): The 3D NumPy array to write to the VTK file.
        reference_grid (vtk.vtkStructuredPoints): The reference VTK grid.
        output_file_name (str): The name of the output VTK file.
        scalar_name (str, optional): The name of the scalar field in the VTK file. Defaults to "scalars".
    """
    try:
        # Ensure the NumPy array is 3D
        if len(np_array.shape) != 3:
            raise ValueError("Input np_array must be a 3D array.")
        
        # Flatten the 3D NumPy array in a way that's consistent with VTK's grid
        flat_np_array = np_array.flatten(order='F')
        
        # Convert the flattened NumPy array to a VTK array
        vtk_array = numpy_to_vtk(flat_np_array, deep=True, array_type=vtk.VTK_FLOAT)
        vtk_array.SetName(scalar_name)
        
        # Create a new VTK grid and set the points and scalar field
        output_grid = vtk.vtkStructuredPoints()
        output_grid.SetDimensions(reference_grid.GetDimensions())
        output_grid.SetOrigin(reference_grid.GetOrigin())
        output_grid.SetSpacing(reference_grid.GetSpacing())
        output_grid.GetPointData().SetScalars(vtk_array)
        
        # Write the VTK grid to a file
        writer = vtk.vtkStructuredPointsWriter()
        writer.SetFileName(output_file_name)
        writer.SetInputData(output_grid)
        writer.Write()
    finally:
        # Explicitly delete objects and collect garbage
        del vtk_array, output_grid, writer
        gc.collect()



import vtk
import numpy as np
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy

def read_and_plot_vtk_z0(file_path, scalar_name, cmap='viridis'):
    """
    Read a VTK file and plot a 2D slice at z=0.

    Parameters:
        file_path (str): The path to the VTK file.
        scalar_name (str): The name of the scalar data to read.
        cmap (str): Colormap to use for the plot. Default is 'viridis'.
    """
    # Read the scalar data from the VTK file
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_path)
    reader.Update()
    data_vtk = reader.GetOutput()
    scalar_data_vtk = data_vtk.GetPointData().GetArray(scalar_name)
    
    if scalar_data_vtk is None:
        raise ValueError(f"The VTK file at {file_path} does not contain scalar data named {scalar_name}.")
    
    # Convert the VTK array to a NumPy array
    scalar_data_np = vtk_to_numpy(scalar_data_vtk)
    
    # Get dimensions of the structured points
    dims = data_vtk.GetDimensions()
    
    # Reshape the 1D NumPy array into a 3D array
    scalar_data_3d = scalar_data_np.reshape(dims, order='F')
    
    # Extract the 2D slice at z=0
    scalar_data_2d = scalar_data_3d[:, :, 0]
    
    # Plot the 2D slice
    plt.imshow(scalar_data_2d.T, origin='lower', cmap=cmap)
    plt.title(f"2D Slice at z=0 of {scalar_name}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label=scalar_name)
    plt.show()


def read_and_plot_vtk_z5(file_path, scalar_name, cmap='viridis'):
    """
    Read a VTK file and plot a 2D slice at z=0.

    Parameters:
        file_path (str): The path to the VTK file.
        scalar_name (str): The name of the scalar data to read.
        cmap (str): Colormap to use for the plot. Default is 'viridis'.
    """
    # Read the scalar data from the VTK file
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_path)
    reader.Update()
    data_vtk = reader.GetOutput()
    scalar_data_vtk = data_vtk.GetPointData().GetArray(scalar_name)
    
    if scalar_data_vtk is None:
        raise ValueError(f"The VTK file at {file_path} does not contain scalar data named {scalar_name}.")
    
    # Convert the VTK array to a NumPy array
    scalar_data_np = vtk_to_numpy(scalar_data_vtk)
    
    # Get dimensions of the structured points
    dims = data_vtk.GetDimensions()
    
    # Reshape the 1D NumPy array into a 3D array
    scalar_data_3d = scalar_data_np.reshape(dims, order='F')
    
    # Extract the 2D slice at z=0
    scalar_data_2d = scalar_data_3d[:, :, 2]
    
    # Plot the 2D slice
    plt.imshow(scalar_data_2d.T, origin='lower', cmap=cmap)
    plt.title(f"2D Slice at z=2 of {scalar_name}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label=scalar_name)
    plt.show()    

# Example usage:
# read_and_plot_vtk_z0("output_G.vtk", "G_scalars")



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import vtk
from vtk.util.numpy_support import vtk_to_numpy

def read_and_plot_vtk_slice(file_path, scalar_name, slice_axis='z', slice_index=0, cmap='viridis', vmin=None, vmax=None):
    """
    Read a VTK file and plot a 2D slice along a specified axis.

    Parameters:
        file_path (str): The path to the VTK file.
        scalar_name (str): The name of the scalar data to read.
        slice_axis (str): Axis along which to take the slice ('x', 'y', or 'z'). Default is 'z'.
        slice_index (int): Index along the specified axis at which to take the slice. Default is 0.
        cmap (str): Colormap to use for the plot. Default is 'viridis'.
        vmin (float): Minimum data value that corresponds to the lower limit of the colormap. If None, uses data minimum.
        vmax (float): Maximum data value that corresponds to the upper limit of the colormap. If None, uses data maximum.
    """
    # Read the scalar data from the VTK file
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_path)
    reader.Update()
    data_vtk = reader.GetOutput()
    scalar_data_vtk = data_vtk.GetPointData().GetArray(scalar_name)
    
    if scalar_data_vtk is None:
        raise ValueError(f"The VTK file at {file_path} does not contain scalar data named {scalar_name}.")
    
    # Convert the VTK array to a NumPy array
    scalar_data_np = vtk_to_numpy(scalar_data_vtk)
    
    # Get dimensions of the structured points
    dims = data_vtk.GetDimensions()
    
    # Reshape the 1D NumPy array into a 3D array
    scalar_data_3d = scalar_data_np.reshape(dims, order='F')
    
    # Extract the 2D slice based on slice_axis and slice_index
    if slice_axis == 'x':
        scalar_data_2d = scalar_data_3d[slice_index, :, :]
    elif slice_axis == 'y':
        scalar_data_2d = scalar_data_3d[:, slice_index, :]
    elif slice_axis == 'z':
        scalar_data_2d = scalar_data_3d[:, :, slice_index]
    else:
        raise ValueError(f"Invalid slice_axis: {slice_axis}. Expected 'x', 'y', or 'z'.")
    
    # Set non-positive values to a small positive number
    epsilon = 1e-13
    scalar_data_2d[scalar_data_2d <= 0] = epsilon
    
    # Use LogNorm for the log scale
    norm = LogNorm(vmin=vmin if vmin else scalar_data_2d.min(),
                   vmax=vmax if vmax else scalar_data_2d.max())
    
    # Plot the 2D slice
    plt.imshow(scalar_data_2d.T, origin='lower', cmap=cmap, norm=norm)
    plt.title(f"2D Slice at {slice_axis}={slice_index} of {scalar_name} (Log Scale)")
    plt.xlabel('x' if slice_axis != 'x' else 'y')
    plt.ylabel('y' if slice_axis != 'y' else 'z')
    cbar = plt.colorbar(label=scalar_name)
    cbar.ax.set_yscale('log')
    plt.show()

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def coarse_grain_numpy_array(np_array, block_size):
    """Coarse grain a 3D numpy array by averaging over blocks of size block_size."""
    # Calculate new dimensions ensuring they are divisible by block_size
    new_dim = tuple((d // block_size) * block_size for d in np_array.shape)

    # Trim the array to make it fit the new dimensions
    trimmed_array = np_array[:new_dim[0], :new_dim[1], :new_dim[2]]

    # Create a view of the array with shape (x, block_size, y, block_size, z, block_size)
    # Then, compute mean over the block_size axes
    return trimmed_array.reshape(trimmed_array.shape[0] // block_size, block_size,
                                 trimmed_array.shape[1] // block_size, block_size,
                                 trimmed_array.shape[2] // block_size, block_size).mean(axis=(1,3,5))


def coarse_grain_vtk_file(input_file, output_file, block_size):
    # Read the VTK file
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(input_file)
    reader.Update()
    input_data = reader.GetOutput()
    
    # Extract scalar data and convert it to numpy array
    original_np_array = vtk_to_numpy(input_data.GetPointData().GetScalars())
    original_dimensions = input_data.GetDimensions()
    
    # Reshape to 3D array
    reshaped_np_array = original_np_array.reshape(original_dimensions, order='F')
    
    # Perform coarse graining on the numpy array
    coarse_np_array = coarse_grain_numpy_array(reshaped_np_array, block_size)
    
    # Write the coarse-grained data to a new VTK file
    # Using the function you provided earlier
    new_spacing = [input_data.GetSpacing()[i] * block_size for i in range(3)]
    reference_grid = vtk.vtkStructuredPoints()
    reference_grid.SetSpacing(new_spacing)
    reference_grid.SetOrigin(input_data.GetOrigin())
    reference_grid.SetDimensions(coarse_np_array.shape)
    write_numpy_to_vtk_3d(coarse_np_array, reference_grid, output_file)

import vtk
import numpy as np

def angle(file_path):
    # Read the VTK file
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(file_path)
    reader.Update()
    
    # Get the output
    data_vtk = reader.GetOutput()
    
    # Get dimensions and spacing
    dimensions = data_vtk.GetDimensions()
    spacing = data_vtk.GetSpacing()
    
    # Calculate the center
    center = [(dimensions[i] - 1) / 2.0 * spacing[i] for i in range(2)]  # For 2D data
    
    # Initialize arrays for coordinates
    x_coords = np.arange(0, dimensions[0]) * spacing[0] - center[0]
    y_coords = np.arange(0, dimensions[1]) * spacing[1] - center[1]
    
    # Create a meshgrid for the coordinates
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Calculate phi
    phi = np.arctan2(Y, X)
    
    return phi
