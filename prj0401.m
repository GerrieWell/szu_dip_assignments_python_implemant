function main()
%说明：主函数包括三个部分：1.中心化及DFT处理
%                          2.DFT逆变换
%                          3.高斯低通滤波
 
%第一部分：中心化、DFT处理 
         ima = imread('/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/Fig4.41(a).jpg');
         imb = shifftmy(ima);
         imaf = fft2(ima);
         imbf = fft2(imb);
         imbf = log(1 + abs(imbf));
         fa = fftshift(imaf); %使用MATLAB自带的移频函数
         fb = log(1 + abs(fa));
         phase = angle(imaf);
         figure
         subplot(2,2,1);
         imshow(ima);
         title('Fig4.41(a)原图');
         
         subplot(2,2,2);
         imshow(imb);
         title('Fig4.41(a)时域*（-1）^(x+y)');
         
         subplot(2,2,3);
         imshow(imaf, []);
         title('Fig4.41(a)频谱图');
         
         subplot(2,2,4);
         imshow(imbf, []);
         title('中心化、对数变换后的频谱图');
         
         figure
         subplot(1,2,1)
         imshow(fb, []);
         title('Fig4.41(a)中心化后的频谱图');
         subplot(1,2,2)
         imshow(phase, []);
         title('Fig4.41(a)相位图');
         figure
         mesh(fb);
         title('Fig4.41(a)中心化后的三维频谱图');
         figure
         mesh(phase);
         title('Fig4.41(a)三维相位图');
%第一部分结束
 
%第二部分：DFT逆变换输出、中心化
        imaif = ifft2(imaf);
        imaifsf = shifftmy(imaif);
        imar = real(imaifsf);
        figure
        subplot(1,3,1);
        imshow(imaif);
        title('FT逆变换结果');
        subplot(1,3,2);
        imshow(imaifsf);
        title('FT逆变换再*（-1）^(x+y)结果');
        subplot(1,3,3);
        imshow(imar);
        title('FT逆变换结果取其实部');
%第二部分结束
 
%第三部分：高斯低通滤波处理
        iml = ima;
        PQ = paddedsize(size(iml));
        [U,V] = dftuv(PQ(1), PQ(2));
        D0 = 0.05*PQ(2);
        F = fft2(ima, PQ(1), PQ(2));
       
        H = exp(-(U.^2 + V.^2)/(2*(D0^2)));
       
        G = dftfilt(iml, H);
        figure
        subplot(2,2,1);
        imshow(iml);
        title('Fig4.41(a)原图');
        subplot(2,2,2);
        HF = fftshift(H);
        imshow(HF,[]);
        title('高斯低通滤波器');
        subplot(2,2,3);
        F = log(1 + abs(fftshift(F)));
        imshow(F,[]);
        title('Fig4.41(a)频谱图');
        subplot(2,2,4);
        imshow(G,[]);
        title('低通滤波后');
        
        figure
        mesh(H);
        title('高斯低通滤波器');
        figure
        mesh(HF);
        title('高斯低通滤波器中心化');
        
        imfz = iml - uint8(G);
        figure
        imshow(imfz, []);
        title('原图减去滤波后的图像');        
%第三部分结束
function imb = shifftmy( ima )
%说明： 图像矩阵中心化处理
%输入参数：
%         ima:待处理图像
%输出参数：
%         imb:输出图像
%
 
[xs,ys]= size(ima);
for i=1 : xs
    for j=1 : ys
        r = (-1).^(i+j); 
        imb(i,j) = ima(i,j).*r;
    end
end
 
end
function PQ = paddedsize(AB, CD, PARAM)
%PADDEDSIZE Computes padded sizes useful for FFT-based filtering. 
%   PQ = PADDEDSIZE(AB), where AB is a two-element size vector,
%   computes the two-element size vector PQ = 2*AB.
%
%   PQ = PADDEDSIZE(AB, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX(AB).
%
%   PQ = PADDEDSIZE(AB, CD), where AB and CD are two-element size
%   vectors, computes the two-element size vector PQ.  The elements
%   of PQ are the smallest even integers greater than or equal to 
%   AB + CD - 1.
%
%   PQ = PADDEDSIZE(AB, CD, 'PWR2') computes the vector PQ such that
%   PQ(1) = PQ(2) = 2^nextpow2(2*m), where m is MAX([AB CD]). 
    
 
 
if nargin == 1
   PQ  = 2*AB;
elseif nargin == 2 & ~ischar(CD)
   PQ = AB + CD - 1;
   PQ = 2 * ceil(PQ / 2);
elseif nargin == 2
   m = max(AB); % Maximum dimension.
   
   % Find power-of-2 at least twice m.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
elseif nargin == 3
   m = max([AB CD]); % Maximum dimension.
   P = 2^nextpow2(2*m);
   PQ = [P, P];
else 
   error('Wrong number of inputs.')
end
function [ U,V ] = dftuv( M, N )
%DFTUV 实现频域滤波器的网格函数
%   Detailed explanation goes here
u = 0:(M - 1);
v = 0:(N - 1);
idx = find(u > M/2); %找大于M/2的数据
u(idx) = u(idx) - M; %将大于M/2的数据减去M
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);      
 
end
function g = dftfilt(f, H)
%DFTFILT Performs frequency domain filtering.
% G = DFTFILT(F, H) filters F in the frequency domain using the
% filter transfer function H. The output, G, is the filtered
% image, which has the same size as F. DFTFILT automatically pads
% F to be the same size as H. Function PADDEDSIZE can be used to
% determine an appropriate size for H.
%
% DFTFILT assumes that F is real and that H is a real, uncentered
% circularly-symmetric filter function. 
 
 
 
% Obtain the FFT of the padded input.
F = fft2(f, size(H, 1), size(H, 2));
 
% Perform filtering. 
g = real(ifft2(H.*F));
 
% Crop to original size.
g = g(1:size(f, 1), 1:size(f, 2));
