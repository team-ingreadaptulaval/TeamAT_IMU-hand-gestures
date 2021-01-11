clear;clc;

format long
X = csvread('feats1.csv');
[q, r] = qr(X)