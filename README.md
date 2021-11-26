# MultiPhaser
Super-cool image processing tool for automatic quantitative analysis of images from microscope 

Before starting the code one needs to install the following packages, compatible with Python 3.6.5 version

py -m pip install   package_name==package_version


1.   numpy==1.14.1

2.   matplotlib==3.0.2

3.   Pillow==5.1.0

4.   scipy==1.0.0

5.   opencv-python==3.4.0.12


The program automatically opens images in given directories and performs auto detection of background (greyscale threshold) and calculates corresponding amount
of secondary phase (area %), saves the data in .CSV, with all the parameters used in the analysis. If more than one picture of given sample (in given directory)
is analyzed, then average is calculated, which is saved in .JSON file, that later on when thousands of images were quantified, serves as database for construction
of plots, histograms, etc. Author: Andraž Kocjan ( more info: taoteck@gmail.com ) in July 2019.



*****************************************************
EXECUTABLE Python file is >>> hea_modeliranje_AI.py
****************************************************





""" ....................... Slovenian language README ................................................................................................................ """

Navodila za inštalacijo programa MultiPhaser v okolju Windows. Avtor: Andraž Kocjan, IMT, Julij 2019.

*************************************************************************
Program je napisan v skriptnem jeziku Python, zato je potrebno najprej inštalirati paket Python 3.6.4 (exe.file v istoimenski mapi), nato pa po spodnjem vrstnem redu še vse pripadajoče pakete.
*************************************************************************

Inštalacija podpornih paketov poteka v Ukaznem pozivu (Command Promt), ki se
zažene z  Iskanje po sistemu Windows (lupa v levem spodnjem kotu) , kamor se vpiše
CMD in klikne na ikono na vrhu. Ko se nam odpre okno Ukaznega poziva lahko najprej preverimo uspešnost inštalacije programa Python 3.6.4 in sicer da vpišemo "py" in pritisnemo Enter, npr.:

C:\Users\uporabnik>py


Če je bila inštalacija uspešna se pojavi:

Python 3.6.4 (v3.6.4:d48eceb, Dec 19 2017, 06:04:45) [MSC v.1900 32 bit (Intel)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>


Z ukazom "py" v Ukaznem pozivu zaženemo Python, zato je pred nadaljevanjem 
inštalacije ostalih paketov potrebno le-tega zapustiti, kar storimo s preprostim ukazom

>>> exit()


kar nas vrne nazaj na

C:\Users\uporabnik>
 

Inštalacija Python paketov (modulov) se vrši s sledečo ukazno vrstico:

py -m pip install   ime_paketa==verzija_paketa


Konkretno, če želimo inštalirati paket "numpy", in sicer verzijo 1.14.1, potem
ukazna vrstica izgleda takole:

C:\Users\uporabnik>py -m pip install numpy==1.14.1


****************************************************************************
V nadaljevanju so navedeni vsa imena paketov skupaj z verzijo, ki jih je potebno inštalirati:

---------------------------------------------------------------

1.   numpy==1.14.1

2.   matplotlib==3.0.2

3.   Pillow==5.1.0

4.   scipy==1.0.0

5.   opencv-python==3.4.0.12








