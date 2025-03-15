1. create conda env at current project folder
   > conda create --prefix ./env python=3.11

    - To activate this environment, use
        > conda activate D:\CausalityProject\Causality\env
    - To deactivate an active environment, use
        > conda deactivate

2. install package
    > python -m pip install --config-settings="--global-option=build_ext" --config-settings="--global-option=-IC:\Program Files\Graphviz\include" --config-settings="--global-option=-LC:\Program Files\Graphviz\lib" pygraphviz
    > pip install -r requirements.txt

    - export the Environment to a YAML File
        > conda env export > environment.yml