# Histone Modification Language Models

This repository contains the code for Chapter 3 of my PhD thesis, entitled Histone Modification Occupancy Prediction Through Language Modelling.

The Hidden Markov Model code is based on [this implementation](https://github.com/TreB1eN/HiddenMarkovModel_Pytorch) of HMMs in Pytorch, but with several bug fixes and various other improvements.

The supervised PCFGs were evaluated using Mark Johnson's [CKY algorithm implementation](http://web.science.mq.edu.au/~mjohnson/Software.htm). The unsupervised PCFGs were built using [Kim et al's](https://github.com/harvardnlp/compound-pcfg) neural and compound PCFG code.
