function imb = shifftmy( ima )
%??? ?????????
%?????
%         ima:?????
%?????
%         imb:????
%
 
[xs,ys]= size(ima);
for i=1 : xs
    for j=1 : ys
        r = (-1).^(i+j); 
        imb(i,j) = ima(i,j).*r;
    end
end
 
end