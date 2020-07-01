# Gradient Temporal-Difference Learning with Regularized Corrections

### [Paper]() | [Documentation](#library-documentation) | [Experiments](#experiments)

- Sina Ghiassian* [[github](https://github.com/sinaghiassian)]
- Andrew Patterson* [[github](https://github.com/andnp) | [website](https://andnp.github.io) | [scholar](https://scholar.google.ca/citations?user=jd2nCqYAAAAJ)]
- Shivam Garg [[github](https://github.com/svmgrg) | [website](https://svmgrg.github.io/)]
- Dhawal Gupta [[github](https://github.com/dhawgupta) | [website](https://dhawgupta.github.io/) | [scholar](https://scholar.google.ca/citations?user=n1Lsp_8AAAAJ)]
- Adam White [[github](https://github.com/amw8) | [website](https://sites.ualberta.ca/~amw8/) | [scholar](https://scholar.google.ca/citations?user=1GqGhcsAAAAJ)]
- Martha White [[github](https://github.com/marthawhite) | [website](https://webdocs.cs.ualberta.ca/~whitem/) | [scholar](https://scholar.google.ca/citations?user=t5zdD_IAAAAJ)]

# Library Documentation
This repo can be installed as a python library using
```bash
pip install git+https://github.com/rlai-lab/regularized-gradienttd.git
```
We provide implementations of the three learning algorithms introduced in our paper:
 * [TDRC](TDRC/TDRC.py) - for non-linear state-value function approximation
 * [DQRC](TDRC/DQRC.py) - for non-linear control using action-value function approximation
 * [DQC](TDRC/DQC.py) - a special case of the DQRC agent where `\beta = 0`.

Each learning algorithm is implemented using [torch]() and optimizes an arbitrary neural network as specified by the user.
blah blah

Here we provide a description of the TDRC algorithm.

Here we provide an example of the usage of the TDRC algorithm.
```python
from TDRC import TDRC

agent = TDRC(parameter_dict)

agent.init()
agent.update(s, a, r, sp)

v = agent.predict(x)
```

Here we provide an example of the usage of the DQRC algorithm.
```python
from TDRC import DQRC

agent = DQRC(parameter_dict)

agent.init()
a = agent.update(s, a, r, sp)

q = agent.predict(x)
a = agent.selectAction(x)
```

# Experiments
Here is a description of the experiments that we include in this readme.
We will provide code to reproduce the results of the linear prediction experiments in the the paper as well as the control experiments on the Mountain Car domain.
In the future we will additionally include the code to reproduce the experiments on the [MinAtar]() environment.

Here are the results in Figure 1 of the paper.
Here is a brief discussion of what they mean and how they were generated.

Here is some brief discussion on how to run the script to generate the results in Figure 1.
We will plot a less pretty version of the plot included in Figure 1.
```bash
cd experiments/prediction
python figure1.py
```

Do the same for Figure 2.

Do the same for Figure 3.

Here is a brief discussion of the control algorithm for Mountain Car.

Here are the non-linear control on Mountain Car results.

Here is a brief discussion of how to run the non-linear control on Mountain Car script.

# Citation

If you find our work useful in your research, please cite:
```

```
