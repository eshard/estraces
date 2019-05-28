pip install -r requirements.txt
sphinx-build -M clean source build
sphinx-build -M html source build
mv build/html ../public/

