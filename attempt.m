clc;
clear;
close all;
imgSetSize=60000;
testSize= 10000;

[imagestrain,labelstrain] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', imgSetSize, 0);
[imagestest,labelstest] = readMNIST('train-images.idx3-ubyte', 'train-labels.idx1-ubyte', testSize, 0);


%Split into training and validation sets
    images50k=imagestrain(:,1:50000);
    labels50k=labelstrain(1:50000,:);
    validimages=imagestrain(:,50001:60000);
    validlabels=labelstrain(50001:60000,:);
    
  %Find prior probabilities 
  p1=zeros(784,4932);
  p2=zeros(784,5678);
  p3=zeros(784,4968);
  p4=zeros(784,5101);
  p5=zeros(784,4859);
  p6=zeros(784,4506);
  p7=zeros(784,4951);
  p8=zeros(784,5175);
  p9=zeros(784,4842);
  p10=zeros(784,4988);
  
 z=zeros(1,10);
i0=0;i1=0;i2=0;i3=0;i4=0;i5=0;i6=0;i7=0;i8=0;i9=0;
 for m=1:50000
     if labels50k(m,:)==0
             z(1)=z(1)+1;
             p1(:,i0+1)=images50k(:,m);
             i0=i0+1;
         elseif labels50k(m,:)==1
             z(2)=z(2)+1;
              p2(:,i1+1)=images50k(:,m);
             i1=i1+1;
         elseif labels50k(m,:)==2
             z(3)=z(3)+1;
              p3(:,i2+1)=images50k(:,m);
             i2=i2+1;
         elseif labels50k(m,:)==3
             z(4)=z(4)+1;
              p4(:,i3+1)=images50k(:,m);
             i3=i3+1;
         elseif labels50k(m,:)==4
             z(5)=z(5)+1;
              p5(:,i4+1)=images50k(:,m);
             i4=i4+1;
         elseif labels50k(m,:)==5
             z(6)=z(6)+1;
              p6(:,i5+1)=images50k(:,m);
             i5=i5+1;
         elseif labels50k(m,:)==6
             z(7)=z(7)+1;
              p7(:,i6+1)=images50k(:,m);
             i6=i6+1;
         elseif labels50k(m,:)==7
             z(8)=z(8)+1;
              p8(:,i7+1)=images50k(:,m);
             i7=i7+1;
         elseif labels50k(m,:)==8
             z(9)=z(9)+1;
              p9(:,i8+1)=images50k(:,m);
             i8=i8+1;
         else 
             z(10)=z(10)+1;
              p10(:,i9+1)=images50k(:,m);
             i9=i9+1;
     end
 end
 
pie=z/50000;

%Find mean and covariance of data points
%%Mean
m1=(1/z(1))*sum(p1,2);
m2=(1/z(2))*sum(p2,2);
m3=(1/z(3))*sum(p3,2);
m4=(1/z(4))*sum(p4,2);
m5=(1/z(5))*sum(p5,2);
m6=(1/z(6))*sum(p6,2);
m7=(1/z(7))*sum(p7,2);
m8=(1/z(8))*sum(p8,2);
m9=(1/z(9))*sum(p9,2);
m10=(1/z(10))*sum(p10,2);
sumc=zeros(784,784);
I=eye(784,784);
%%Covariance
for q=1:z(1)
    sumc=sumc+(p1(:,q)-m1)*(p1(:,q)-m1)';
c1=(1/(z(1)-1))*sumc;
C1=c1+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(2)
    sumc=sumc+(p2(:,q)-m2)*(p2(:,q)-m2)';
c2=(1/(z(2)-1))*sumc;
C2=c2+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(3)
    sumc=sumc+(p3(:,q)-m3)*(p3(:,q)-m3)';
c3=(1/(z(3)-1))*sumc;
C3=c3+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(4)
    sumc=sumc+(p4(:,q)-m4)*(p4(:,q)-m4)';
c4=(1/(z(4)-1))*sumc;
C4=c4+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(5)
    sumc=sumc+(p5(:,q)-m5)*(p5(:,q)-m5)';
c5=(1/(z(5)-1))*sumc;
C5=c5+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(6)
    sumc=sumc+(p6(:,q)-m6)*(p6(:,q)-m6)';
c6=(1/(z(6)-1))*sumc;
C6=c6+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(7)
    sumc=sumc+(p7(:,q)-m7)*(p7(:,q)-m7)';
c7=(1/(z(7)-1))*sumc;
C7=c7+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(8)
    sumc=sumc+(p8(:,q)-m8)*(p8(:,q)-m8)';
c8=(1/(z(8)-1))*sumc;
C8=c8+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(9)
    sumc=sumc+(p9(:,q)-m9)*(p9(:,q)-m9)';
c9=(1/(z(9)-1))*sumc;
C9=c9+(0.5)*I;
end
sumc=zeros(784,784);
for q=1:z(10)
    sumc=sumc+(p10(:,q)-m10)*(p10(:,q)-m10)';
c10=(1/(z(10)-1))*sumc;
C10=c10+(0.5)*I;
end

%step5:Testing Phase
post=zeros(10,10000);
counter=0;
error=0;
for q=1:10000;
t1=mvnpdf(imagestest(:,q),m1,C1);
t2=mvnpdf(imagestest(:,q),m2,C2);
t3=mvnpdf(imagestest(:,q),m3,C3);
t4=mvnpdf(imagestest(:,q),m4,C4);
t5=mvnpdf(imagestest(:,q),m5,C5);
t6=mvnpdf(imagestest(:,q),m6,C6);
t7=mvnpdf(imagestest(:,q),m7,C7);
t8=mvnpdf(imagestest(:,q),m8,C8);
t9=mvnpdf(imagestest(:,q),m9,C9);
t10=mvnpdf(imagestest(:,q),m10,C10);

Q=[t1;t2;t3;t4;t5;t6;t7;t8;t9;t10];
[val,ind]=max(Q);
e=labelstest(q)-(ind-1);
if(e~=0)
    post(:,q)=Q;
    counter=counter+1;
    correct_digit=labelstest(q);
    prediction=ind-1;
    fprintf('Correct digit is %d\nrecognised digit is %d\n',correct_digit,prediction);
    
end
error=counter/100;
end

disp('Error percentage');
disp(error);





