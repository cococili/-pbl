im = imread('hi-bazaar-alley.jpg');
im=rgb2gray(im);
figure;
subplot(131),imshow(im);title('原图空间域');
%绘制原图频谱
F0=fft2(im);%这里要注意一定加一个反转，不加虽然不会错，但是频谱图效果不好
F00=log(abs(fftshift(F0))+1);
subplot(132);imshow(F00,[]);title('原图幅度谱');
subplot(133);imshow(angle(fftshift(F0)),[]);title('原图相位谱');


% 1）生成含有高斯噪声、椒盐噪声的图像
% imnoise 是表示添加噪声污染一幅图像，叫做噪声污染图像函数
im_noise_salt = imnoise(im,'salt & pepper'); % 加入椒盐噪声
im_noise_gaussian = imnoise(im,'gaussian'); % 加入高斯噪声
im_noise_poisson = imnoise(im,'poisson'); % 加入泊松噪声


%绘制椒盐幅度谱
F1=fft2(im_noise_salt);
F11=log(abs(fftshift(F1))+1);
figure;
subplot(3,3,1);imshow(F11,[]);title('加入椒盐噪声后的图像幅度谱');
subplot(3,3,4);imshow(angle(fftshift(F1)),[]);title('加入椒盐噪声后的图像相位谱');

%绘制高斯幅度谱
F2=fft2(im_noise_gaussian);
F22=log(abs(fftshift(F2))+1);
subplot(3,3,2);imshow(F22,[]);title('加入高斯噪声后的图像幅度谱');
subplot(3,3,5);imshow(angle(fftshift(F2)),[]);title('加入高斯噪声后的图像相位谱');

%绘制泊松幅度谱
F3=fft2(im_noise_poisson);
F33=log(abs(fftshift(F3))+1);
subplot(3,3,3);imshow(F33,[]);title('加入泊松噪声后的图像幅度谱');
subplot(3,3,6);imshow(angle(fftshift(F3)),[]);title('加入泊松噪声后的图像相位谱');

%时域图
subplot(3,3,7),imshow(im_noise_salt);title('加入椒盐噪声后的图像空间域图');
subplot(3,3,8),imshow(im_noise_gaussian);title('加入高斯噪声后的图像空间域图');
subplot(3,3,9),imshow(im_noise_poisson);title('加入泊松噪声后的图像空间域图');



% 2）使用均值滤波分别对高斯噪声、椒盐噪声的图像进行滤波
% fspecial函数 用来生成滤波器（也叫算子）的函数
% h = fspecial(type)  h = fspecial(type，para) 
% 使用type参数来指定滤波器的种类，使用para来对具体的滤波器种类添加额外的参数信息。h就是生成的滤波器。
n=1; m=2*n+1;
A = fspecial('average',m); % 生成系统自带3×3滤波器

% filter2 - 二维数字滤波器 
% Y = filter2(H,X)  根据矩阵 H 中的系数，对数据矩阵 X 应用有限脉冲响应滤波器。
% 进行滤波并显示图像
im_filtered1 = filter2(A,im_noise_salt);
im_filtered2 = filter2(A,im_noise_gaussian);
im_filtered3 = filter2(A,im_noise_poisson);
figure;
subplot(331),imshow(im_noise_salt);title('加入椒盐噪声后的图像');
subplot(332),imshow(im_noise_gaussian);title('加入高斯噪声后的图像');
subplot(333),imshow(im_noise_poisson);title('加入泊松噪声后的图像');

subplot(334),imshow(uint8(im_filtered1));title('椒盐噪声图像进行均值滤波后的图像');
subplot(335),imshow(uint8(im_filtered2));title('高斯噪声图像进行均值滤波后的图像');
subplot(336),imshow(uint8(im_filtered3));title('泊松噪声图像进行均值滤波后的图像');

% 3）使用中值滤波分别对高斯噪声、椒盐噪声的图像进行滤波
% 定义邻域尺寸
n1 = 2; m1 = 2*n1+1;
n2 = 2; m2 = 2*n2+1;

% medfilt2函数用于执行二维中值滤波，使用方法如下：
% B = medfilt2(A, [m n]) B = medfilt2(A)
% 其中[m n]表示邻域块的大小，默认值为[3 3]。 b=medfilt2(a,[m,n]);
% b是中值滤波后的图象矩阵，a是原图矩阵，m和n是处理模版大小，默认3×3。
im_filtered11 = medfilt2(im_noise_salt,[m1,m2]);
im_filtered22 = medfilt2(im_noise_gaussian,[m1,m2]);
im_filtered33 = medfilt2(im_noise_poisson,[m1,m2]);
subplot(337),imshow(im_filtered11);title('椒盐噪声图像进行中值滤波后的图像');
subplot(338),imshow(im_filtered22);title('高斯噪声图像进行中值滤波后的图像');
subplot(339),imshow(im_filtered33);title('泊松噪声图像进行中值滤波后的图像');



%绘制椒盐均值去噪幅度谱
F1_=fft2(im_filtered1);
F11_=log(abs(fftshift(F1_))+1);
figure;
subplot(4,3,1);imshow(F11_,[]);title('椒盐均值去噪声后的图像幅度谱');
subplot(4,3,4);imshow(angle(fftshift(F1_)),[]);title('椒盐均值去噪声后的图像相位谱');

%绘制高斯均值去噪幅度谱
F2_=fft2(im_filtered2);
F22_=log(abs(fftshift(F2_))+1);
subplot(4,3,2);imshow(F22_,[]);title('高斯均值去噪声后的图像幅度谱');
subplot(4,3,5);imshow(angle(fftshift(F2_)),[]);title('高斯均值去噪声后的图像相位谱');


%绘制泊松均值去噪幅度谱
F3_=fft2(im_filtered3);
F33_=log(abs(fftshift(F3_))+1);
subplot(4,3,3);imshow(F33_,[]);title('泊松均值去噪声后的图像幅度谱');
subplot(4,3,6);imshow(angle(fftshift(F3_)),[]);title('泊松均值去噪声后的图像相位谱');


%绘制椒盐中值去噪幅度谱
F1_1=fft2(im_filtered11);
F11_1=log(abs(fftshift(F1_1))+1);
subplot(4,3,7);imshow(F11_1,[]);title('椒盐中值去噪声后的图像幅度谱');
subplot(4,3,10);imshow(angle(fftshift(F1_1)),[]);title('椒盐中值去噪声后的图像相位谱');

%绘制高斯中值去噪幅度谱
F2_1=fft2(im_filtered22);
F22_1=log(fftshift(F2_1)+1);
subplot(4,3,8);imshow(F22_1,[]);title('高斯中值去噪声后的图像幅度谱');
subplot(4,3,11);imshow(angle(fftshift(F2_1)),[]);title('高斯中值去噪声后的图像相位谱');

%绘制泊松中值去噪幅度谱
F3_1=fft2(im_filtered33);
F33_1=log(abs(fftshift(F3_1))+1);
subplot(4,3,9);imshow(F33_1,[]);title('泊松中值去噪声后的图像幅度谱');
subplot(4,3,12);imshow(angle(fftshift(F3_1)),[]);title('泊松中值去噪声后的图像相位谱');


%评估椒盐均值
[h,w]=size(im_filtered1);
imgn=imresize(im_filtered1,[floor(h/2) floor(w/2)]);
imgn=imresize(imgn,[h w]);
img=double(im_filtered1);
imgn=double(imgn);

B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES_salt_mean=sum(sum((img-imgn).^2))/(h*w)   %均方差
PSNR_salt_mean=20*log10(MAX/sqrt(MES_salt_mean))        %峰值信噪比


%评估椒盐中值
[h,w]=size(im_filtered11);
imgn=imresize(im_filtered11,[floor(h/2) floor(w/2)]);
imgn=imresize(imgn,[h w]);
img=double(im_filtered11);
imgn=double(imgn);

B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES_salt_Median=sum(sum((img-imgn).^2))/(h*w)   %均方差
PSNR_salt_Median=20*log10(MAX/sqrt(MES_salt_Median))        %峰值信噪比

%评估高斯均值
[h,w]=size(im_filtered2);
imgn=imresize(im_filtered2,[floor(h/2) floor(w/2)]);
imgn=imresize(imgn,[h w]);
img=double(im_filtered2);
imgn=double(imgn);

B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES_gaussian_mean=sum(sum((img-imgn).^2))/(h*w)   %均方差
PSNR_gaussian_mean=20*log10(MAX/sqrt(MES_gaussian_mean))        %峰值信噪比


%评估高斯中值
[h,w]=size(im_filtered22);
imgn=imresize(im_filtered22,[floor(h/2) floor(w/2)]);
imgn=imresize(imgn,[h w]);
img=double(im_filtered22);
imgn=double(imgn);

B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES_gaussian_Median=sum(sum((img-imgn).^2))/(h*w)   %均方差
PSNR_gaussian_Median=20*log10(MAX/sqrt(MES_gaussian_Median))        %峰值信噪比


%评估泊松均值
[h,w]=size(im_filtered3);
imgn=imresize(im_filtered3,[floor(h/2) floor(w/2)]);
imgn=imresize(imgn,[h w]);
img=double(im_filtered3);
imgn=double(imgn);

B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES_poisson_mean=sum(sum((img-imgn).^2))/(h*w)   %均方差
PSNR_poisson_mean=20*log10(MAX/sqrt(MES_poisson_mean))        %峰值信噪比


%评估泊松中值
[h,w]=size(im_filtered33);
imgn=imresize(im_filtered33,[floor(h/2) floor(w/2)]);
imgn=imresize(imgn,[h w]);
img=double(im_filtered33);
imgn=double(imgn);

B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES_poisson_Median=sum(sum((img-imgn).^2))/(h*w)   %均方差
PSNR_poisson_Median=20*log10(MAX/sqrt(MES_poisson_Median))        %峰值信噪比
