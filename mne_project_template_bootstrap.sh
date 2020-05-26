PKG_NAME=mne-foo
PYTHON_NAME=mnefoo
GH_NAME=my_github_user

unameOut="$(uname -s)"

if [[ "${unameOut}" == "Linux" ]]; then
        git grep -l 'mnetemplate' | xargs sed -i 's/mnetemplate/'"${PYTHON_NAME}"'/g'
        git grep -l 'mne-template' | xargs sed -i 's/mne-template/'"${PKG_NAME}"'/g'
        sed -i 's/mne-tools/'"${GH_NAME}"'/g' README.rst
        sed -i 's/mne-project-template/'"${PKG_NAME}"'/g' README.rst
        sed -i 's/mne-tools/'"${GH_NAME}"'/g' setup.py
        sed -i 's/mne-project-template/'"${PKG_NAME}"'/g' setup.py
else
    git grep -l 'mnetemplate' | xargs sed -i ' ' -e 's/mnetemplate/'"${PYTHON_NAME}"'/g'
    git grep -l 'mne-template' | xargs sed -i ' ' -e 's/mne-template/'"${PKG_NAME}"'/g'
    sed -i ' ' -e 's/mne-tools/'"${GH_NAME}"'/g' README.rst
    sed -i ' ' -e 's/mne-project-template/'"${PKG_NAME}"'/g' README.rst
    sed -i ' ' -e 's/mne-tools/'"${GH_NAME}"'/g' setup.py
    sed -i ' ' -e 's/mne-project-template/'"${PKG_NAME}"'/g' setup.py
fi
mv mnetemplate ${PYTHON_NAME}
