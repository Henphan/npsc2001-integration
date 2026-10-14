# NPSC2001 | Phase 2
## The integration script for Giulliano's pipeline and AutoMorph
This repository contains the script which integrates AutoMorph and the fibre-enhancement pipeline. This script allows the user to simply place retinal data inside images/ and not have to worry about their vessel masks.
## Important notes:
- The automation.py program was initially designed to only work with a very specific dataset, with a very specific structure. Various modfications would need to be made to generalise this script to a different dataset.
## How do I run the code?
1. Install the Python packages
```
pip install requirements.txt
```
2. Clone AutoMorph
```
gh repo clone https://github.com/rmaphoh/AutoMorph
```
3. Set the paths at the start of 'automation.py' to the correct locations for you.
4. Run automation.py, 
```
python automation.py
```
5. Run the fibre-enhancement pipeline:
```
python preproccessing.py
```


