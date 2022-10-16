function [output,neuron] = NeuralNetwork(inp,n_Neuron,w,b)
%% Initialize Neural Network
neuron=cell(1,length(n_Neuron));
for i=1:length(n_Neuron)
    neuron{1,i}=zeros(n_Neuron(1,i),1);
end

%% Feedforward Propagation
neuron{1,1}=inp;
for k=1:length(w)
    neuron{1,k+1}=w{1,k}*neuron{1,k}+b{1,k};
    neuron{1,k+1}=1./(1+exp(-neuron{1,k+1}));
end

%% Output
output=neuron{1,length(n_Neuron)};
end