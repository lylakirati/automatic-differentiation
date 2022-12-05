# team20-forwardAD

### Members: Isaac Lee, Tom Zhang, Mina Lee, Lyla Kiratiwudhikul, Youngseon Park
![example workflow](https://code.harvard.edu/CS107/team20/actions/workflows/test.yml/badge.svg)
![example workflow](https://code.harvard.edu/CS107/team20/actions/workflows/coverage.yml/badge.svg)

### Introduction
---

 Our tool, forwardAD, uses automatic differentiation (AD) in forward mode to compute derivatives of functions ranging from simple to complex functions. Unlike conventional methods for evaluating derivatives (e.g., symbolic derivatives, finite differences) that are computationally expensive or lack accuracy/stability, AD enables us to calculate derivatives with machine precision without compromising accuracy and stability. We believe that this tool will be used in a wide range of applications where fast and accurate differential calculations, especially optimization, are required.

### Documentation
---

For documentation and how-to-use, please reiew our [documentation notebook](https://code.harvard.edu/CS107/team20/blob/main/docs/documentation.md).

For final presentation slides, please follow this [link](https://docs.google.com/presentation/d/1QTp1TgBgD-8IoDuCckQiYSAcb674EGMxv8NQPq3YPPE/edit?usp=sharing)

### Modules
---
We have three modules in our package `forwardFD`.

* `forwardFD` : a module that calculates derivatives by AD.
* `dualNumber` : a module that defines an object consisting of scalar and derivative values at each node in AD.
* `elementary`: a module that consists of all basic operations and elementary functions.

### Broader Impact and Inclusivity Statement

 In a dynamic world, the ability to track the change is essential in most academic fields. Our tool, forwardAD, uses automatic differentiation (AD) in forward mode to compute derivatives of functions ranging from simple to complex functions. Unlike conventional methods for evaluating derivatives (e.g., symbolic derivatives, finite differences) that are computationally expensive or lack accuracy/stability, AD enables us to calculate derivatives with machine precision without compromising accuracy and stability. We believe that this tool will be used in a wide range of applications where fast and accurate differential calculations, especially optimization, are required.
The potential positive impact will be a contribution to energy savings by calculating complex derivatives with less computational energy. While AI and ML research improves human life, training an advanced AI or ML models takes time, money, high-quality data, and a huge amount of energy. Our tools, with their ability to compute efficiently with less energy, will contribute to the ongoing energy-wasting problem in computer science research, and ultimately have a positive impact on the climate. The possible negative impact is misuse of our tool by students who are just starting to learn calculus. Because our tool is user-friendly, it can be a good tool for students who do their homework and don't want to spend time figuring out questions personally. To prevent this potential issue, we will release an educational package of forwardAD with visual explanations of the calculations.

While our tool is user-friendly, it is developed under the assumption that users of our package have a basic familiarity with python, calculus, and mathematical terminologies in English. It will exclude a vast portion of our community who do not have these fundamental abilities. To make our package more inclusive, we will launch a web-based extension of our package in which any user can enjoy our tool by simply enter their functions of interest and values. 

