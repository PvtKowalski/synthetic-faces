# synthetic-data

To train neural nets download __data_samples.zip__ from https://drive.google.com/file/d/1e810BI-WemXoT6Gp7wSB-1NgIsv2qlK5/view?usp=sharing and unpack it into __synthdata__ drectory.

To use your own data prepare your own csv files with paths to samples and ground truth.

In order to generate synthetic data:

1. Generate human mesh with Makehuman scripts. You have to launch MH and
  * drop custom random plugin into plugins folder;
  * edit geometries folder structure.
  * copypaste script into its scripting module.
  * launch it from execute tab.
2. Render it with Blender scripts. You have to put absolute paths in params.py and beginning of render_script.py.
