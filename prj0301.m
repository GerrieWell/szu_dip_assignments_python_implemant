         ima = imread('/Users/gerrie/Documents/szu/course/DIP_assignments/Labimages/Fig3.08(a).jpg');
         c=0.9;
         imad=convlog(ima,c);
         figure
         subplot(1,3,1);
         imshow(ima);
         title('?????');
         subplot(1,3,2);
         imshow(imad);
         title(['???????? ???? c=' num2str(c)]);
         
         d = 0.35;
         k = 1.2;
         imam = convmi(ima,d,k);
         subplot(1,3,3);
         imshow(imam);
         title(['???????? ???? ?=' num2str(d)   ','  'k=' num2str(k)]);
         figure
         subplot(2,2,1);
         imshow(ima);
         title('?????');
         
         subplot(2,2,2);
         eq=histeq(ima);
         imshow(eq);
         title('?????????');
         
         subplot(2,2,3);
         imhist(ima);
         title('????????');
         
         subplot(2,2,4);
         imhist(eq);
         title('????????');
        