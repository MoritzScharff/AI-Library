%1. Example: Neurales Network / Machine Learning
%3 Input Parameters (0-1), if Sum > 1 --> Output = 1
%Sigmoid as Activation Function

%% Initialization
clear;
close all;
clc;

%% Parameter Definition
n_Neuron=[3,3,2];                                                           %Number of Neurons a Layer
w=cell(1,length(n_Neuron)-1);                                               %Weight Matrices
b=cell(1,length(n_Neuron)-1);                                               %Bias Vectors

inp_training=rand(3,1000);                                                   %Input Data for Training
outp_training=sum(inp_training);
outp_training(outp_training<1)=0;
outp_training(outp_training~=0)=1;
outp_training=[outp_training;~outp_training];                               %Output Data for Training

learning_rate=0.2;                                                          %Scalar Factor of Gradient Descent

%% Initial Guess - Random Numbers
for i=1:length(w)
    w{1,i}=rand(n_Neuron(i+1),n_Neuron(i));
    b{1,i}=rand(n_Neuron(i+1),1);
end

%% Training Neural Network
n=50;
counter=zeros(1,n);
for j=1:n
    for i=1:size(inp_training,2)
        %Feedforward Propagation
        [~,neuron]=NeuralNetwork(inp_training(:,i),n_Neuron,w,b);
        e=neuron{1,length(n_Neuron)}-outp_training(:,i);
        e_mse=mean(e.^2);

        %Check Performance
        [~,q1]=max(neuron{1,length(n_Neuron)});
        [~,q2]=max(outp_training(:,i));
        if q1==q2
            counter(j)=counter(j)+1;
        end

        %Back Propagation
        del=e;
        w_h=eye(length(del),length(del));
        for k=length(w):-1:1
            del=w_h'*del.*neuron{1,k+1}.*(1-neuron{1,k+1});
            w{1,k}=w{1,k}-learning_rate*del*neuron{1,k}';
            b{1,k}=b{1,k}-learning_rate*del;
            w_h=w{1,k};
        end
    end
    counter(j)=counter(j)/size(inp_training,2)*100;
    plot(counter(1:j))
    set(gca,'xlim',[0 n],'ylim',[0 100]);
    title(['Matching Rate: ' num2str(counter(j)) '%'])
    xlabel('Training Iteration');
    ylabel('Matching Rate');
    drawnow;
end