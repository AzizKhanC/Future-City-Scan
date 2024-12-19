
# conda update -n base -c defaults conda
conda create -n crp python=3.8
conda activate crp

cd C:\Users\Aziz\Dropbox\CRP\FCS\scripts
# https://opensourceoptions.com/blog/how-to-install-gdal-with-anaconda/
# conda install -c conda-forge gdal
pip install -r "requirements.txt"

# pip freeze > requirements.txt

pip install --user ipykernel
python -m ipykernel install --user --name=crp
conda install -c conda-forge ipywidgets
