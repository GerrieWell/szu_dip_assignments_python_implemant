function main()
%
%第一部分：使用对数函数扩展图像灰度级
%功能： 将输入图像中较窄的灰度值映射为输出较宽的灰度值
% 
         ima = imread('/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/Fig3.08\');
         c=0.9;
         imad=convlog(ima,c);
         figure
         subplot(1,3,1);
         imshow(ima);
         title('原输入图像');
         subplot(1,3,2);
         imshow(imad);
         title(['对数变换后的图像 比例系数 c=' num2str(c)]);
         
         d = 0.35;
         k = 1.2;
         imam = convmi(ima,d,k);
         subplot(1,3,3);
         imshow(imam);
         title(['幂律变换后的图像 指数系数 γ=' num2str(d)   ','  'k=' num2str(k)]);
         figure
         subplot(2,2,1);
         imshow(ima);
         title('原输入图像');
         
         subplot(2,2,2);
         eq=histeq(ima);
         imshow(eq);
         title('直方图均衡化后的图');
         
         subplot(2,2,3);
         imhist(ima);
         title('原输入图像直方图');
         
         subplot(2,2,4);
         imhist(eq);
         title('均衡化后的直方图');
        
         figure
         [count,eqcount] = imhistmy(ima);
         x = 0:1:255;
         subplot(1,2,1);
         stem(x,count,'.');
         title('自制直方图');
         subplot(1,2,2);
         stem(x,eqcount,'.');
         title('自制均衡直方图');
         
         imeq = histeqmy(ima,eqcount);
         figure
         subplot(1,2,1);
         imshow(imeq);
         title('自制均衡直方图处理后的图像');
         subplot(1,2,2);
         [count,eqcount] = imhistmy(imeq);
         x = 0:1:255;
         stem(x,count,'.');
         title('自制直方图均衡化后的图像直方图');

function imaz = convlog( ima,c)
%convlog 使用对数变换扩展灰度级
%   输入参数：
%          ima:输入图像
%            c:比例因子
 
%   输出参数：
%          imaz:使用对数变换扩展后的图像
 
         [xs,ys] = size(ima);
         im=double(ima);
         imab = double(ones(xs,ys)); 
         imad = imab + im;
         imad = double(c*log(imad));
         imaz = uint8(imad/log(256)*255);
 
end
 
function imaz = convmi( ima ,d,c )
%convmi 使用幂律变换扩展灰度级
%   输入参数：
%          ima:输入图像
%            c:比例因子
%            d:指数
 
%   输出参数：
         im=double(ima);
         im = double(im/255);
         imad = im.^d;
         imad = imad*c;
         imaz = uint8(imad*255);
end
 function [count,eqcount] = imhistmy( ima )
%imhist 自制直方图程序
%   此处显示详细说明
%   输入参数：ima 图像
%   输出参数：
%         count:原图直方图统计量
%         eqcount:均衡化直方图统计量
%
[xs,ys]= size(ima);
count = zeros(1,256);
eqcount = zeros(1,256);
for i=1:xs
    for j=1:ys
        for gray =0:255
            if ima(i,j)==gray
                count(gray+1)=count(gray+1)+1;
            end
        end
    end
end
 count = count /xs/ys;  
for j=1:256
    for k=1:j
        eqcount(j)=count(k)+eqcount(j);
    end
end
end

function imb  = histeqmy( ima,count )
%histeqmy 此处显示有关此函数的摘要
%   输入参数：
%           ima:待处理图像
%           count:直方图均衡函数
%   输出参数：
%           imb:均衡后输出图像
[xs,ys]= size(ima);
imb=zeros(xs,ys);
for i=1:256
newt(1,i)=floor(254*count(1,i)+0.5);
end
%计算直方图均衡后的新图
for x=1:xs
    for y=1:ys
      imb(x,y+1)=newt(1,ima(x,y)+1);
    end
end
imb = uint8(imb);       
end 
       
