clear; close all; clc;
data = load ('PiiishingData.txt');
x= data(1:392,1:9);
y = data(1:392,10);
tree = fitctree(x,y);
test = predict (tree,[-1,0,-1,-1,-1,0,1,-1,0]);
fprintf ('Test = %d\n', test);
view (tree,'mode','graph');