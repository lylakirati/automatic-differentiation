# Automatic Differentation

**Authors:** Isaac Lee, Tom Zhang, Mina Lee, Lyla Kiratiwudhikul, Youngseon Park

## Introduction

Our software, team20ad, implements automatic differentiation (AD) in forward and reverse modes for computing derivatives of given functions. Unlike conventional methods for evaluating derivatives (e.g., symbolic derivatives, finite differences) that are computationally expensive or lack accuracy/stability, AD enables us to calculate derivatives with machine precision without compromising accuracy and stability. Automatic differentiation can be used in a wide range of applications where fast and accurate differential calculations, especially optimization, are required.

## Documentation

For documentation and how-to-use, please review our [documentation notebook](https://github.com/lylakirati/automatic-differentiation/blob/main/docs/documentation.md).

For final presentation slides, please follow this [link](https://docs.google.com/presentation/d/1QTp1TgBgD-8IoDuCckQiYSAcb674EGMxv8NQPq3YPPE/edit?usp=sharing)

## Modules

There are five modules in our package `team20ad`.

* `forwardAD` : calculates derivatives by traversing the chain rule from inside to outside (forward mode automatic differentiation).
* `reverseAD` : calculates derivatives by traversing the chain rule from outside to inside (reverse mode automatic differentiation).
* `wrapperAD` : a encapsulation that the user can specify the mode as forwardAD or reverseAD. If the mode is not specified, it automatically determines which mode to use based on the number of independent variables and the number of functions to differentiate.
* `dualNumber` : defines an object consisting of scalar and derivative values at each node in AD.
* `elementary`: defines all basic operations and elementary functions.


## Broader Impact and Inclusivity Statement

 In a dynamic world, the ability to track change is essential in most academic fields. Our tool, team20ad, uses automatic differentiation (AD) in forward mode to compute derivatives of functions ranging from simple to complex functions. Unlike conventional methods for evaluating derivatives (e.g., symbolic derivatives, finite differences) that are computationally expensive or lack accuracy/stability, AD enables us to calculate derivatives with machine precision without compromising accuracy and stability. We believe that this tool will be used in a wide range of applications where fast and accurate differential calculations, especially optimization, are required.

The potential positive impact will be a contribution to energy savings by calculating complex derivatives with less computational energy. While AI and ML research improves human life, training advanced AI or ML models takes time, money, high-quality data, and a huge amount of energy. Our tools, with their ability to compute efficiently with less energy, will contribute to the ongoing energy-wasting problem in computer science research, and ultimately have a positive impact on the environment. The possible negative impact is the misuse of our tool by students who are just starting to learn calculus. Because our tool is user-friendly, it can be a good tool for students who do their homework and don't want to spend time figuring out questions personally. To prevent this potential issue, we will release an educational package of team20ad with visual explanations of the calculation process.

While our tool is user-friendly, it is developed under the assumption that users of our package have a basic familiarity with python, calculus, and mathematical terminologies in English. It will exclude a vast portion of our community who do not have these fundamental abilities. To make our package more inclusive, we plan on launching a web-based extension of our package in which any user can enjoy our tool by simply entering their functions of interest and values. 


## Citation

```bibtex
@misc{kiratiwudhikul-auto-diff,
  author = {Isaac Lee and Tom Zhang and Mina Lee and Lyla Kiratiwudhikul and Youngseon Park},
  title = {Automatic Differentiation (team20ad)},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lylakirati/automatic-differentiation}}
}
```

