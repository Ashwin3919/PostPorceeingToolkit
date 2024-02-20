# PostPorceeingToolkit
python3 Scripts to Post-Porocces the NR Simualtios produced with BAM Code

Fuctions: 

The provided code contains several functions, each designed for specific operations involving VTK files and NumPy arrays. Here's a list of the functions along with a brief description of their purpose:

1. **`read_vtk_to_numpy(file_path, scalar_name)`**: Reads scalar data from a VTK file and converts it to a 3D NumPy array. It also returns the VTK structured points data.

2. **`write_numpy_to_vtk_3d(np_array, reference_grid, output_file_name, scalar_name="scalars")`**: Writes a 3D NumPy array to a VTK file as a scalar field, using a reference VTK grid for dimensions, origin, and spacing.

3. **`read_and_plot_vtk_z0(file_path, scalar_name, cmap='viridis')`**: Reads a VTK file, converts the specified scalar data to a NumPy array, and plots a 2D slice at z=0 using matplotlib.

4. **`read_and_plot_vtk_z5(file_path, scalar_name, cmap='viridis')`**: Similar to `read_and_plot_vtk_z0`, but plots a 2D slice at z=5. Note: The function name might be misleading, as it suggests a fixed slice index, but it's actually a flexible parameter.

5. **`read_and_plot_vtk_slice(file_path, scalar_name, slice_axis='z', slice_index=0, cmap='viridis', vmin=None, vmax=None)`**: Reads a VTK file and plots a 2D slice along a specified axis ('x', 'y', or 'z') at a given index, with options for colormap, and minimum and maximum values for data normalization.

6. **`coarse_grain_numpy_array(np_array, block_size)`**: Applies coarse-graining to a 3D NumPy array by averaging over blocks of a specified size, reducing the array's resolution.

7. **`coarse_grain_vtk_file(input_file, output_file, block_size)`**: Reads a VTK file, extracts scalar data, performs coarse-graining using `coarse_grain_numpy_array`, and writes the reduced data back to a new VTK file.

8. **`angle(file_path)`**: Reads a VTK file and calculates the angular coordinate (phi) for each point in a 2D plane, relative to the center of the plane.

