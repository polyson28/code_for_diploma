# code_for_diploma
The repository contains the code for diploma paper **"MACHINE LEARNING APPLICATIONS TO CALIBRATIONS OF VOLATILITY MODELS"**. 

## Abstract 
Volatility is a crucial component for risk measurement. There has been developed numerous techniques for volatility estimation. Nevertheless, traditional methods for volatility model estimation often face various challenges. For instance, they often rely on unrealistic assumptions and require lots of computational time and resources. The application of deep learning for volatility estimation aims to address the limitations of conventional methods. In specific, neural networks approximate the implied volatility surface by learning the dependencies in the inverse Black-Scholes formula.  
Our paper is dedicated to calibration of volatility models using machine learning techniques. We extend the work of previous researchers and propose a new approach to model calibration. In particular, we cluster the data points based on moneyness and train several neural networks independently in each cluster.  
The results indicate that the proposed approach leads to slight improvement in terms of the ability of the model to learn sophisticated non-linear relations and to generalize well to unseen data. However, the model experiences a slight overfit, that should be accounted for in the future research. 

## The structure of the repository
All the necessary functions are defined in the file *model.py*. The ouputs are illustrated in folder *outputs*. 