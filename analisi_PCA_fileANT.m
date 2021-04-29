clear all
clc
close all
quat1_1=[];quat1_2=[];quat1_3=[];quat1_4=[];
quat2_1=[];quat2_2=[];quat2_3=[];quat2_4=[];
quat3_1=[];quat3_2=[];quat3_3=[];quat3_4=[];
i_1=0;
l_1=1;
var_1=0;
stato_1=0;
i_2=0;
l_2=1;
var_2=0;
stato_2=0;
i_3=0;
l_3=1;
var_3=0;
stato_3=0;
filename='testrespiroseduto.txt';
fid=fopen(filename); %opens the file, filename, for binary read access, and returns an integer file identifier equal to or greater than 3
tline = fgetl(fid); %returns the next line as a character vector of the specified file, removing the newline characters
tlines = cell(0,1); %initializing tlines 0x1
%inserting filename in tlines (?)
%while tline is a char repeat the loop
while ischar(tline) % returns true if tline is a character array and false otherwise
tlines{end+1,1} = tline; %update tlines with a new row that is tline
tline = fgetl(fid); %returns the next line of the specified file, removing the newline characters
end
fclose(fid); %closes an open file
%tlines=string(tlines)
tlines=string(tlines); %converts a scalar to a string array of words
k=contains(tlines,'Rx:'); %returns 1 (true) if tlines contains the specified pattern Rx:, and returns 0 (false) otherwise
%k is a 32618x1 of 0 and 1 (where Rx is present)
t=tlines(k); %values only of Rx (14807x1)
t=erase(t,'Rx:'); %deletes all occurrences of Rx: in t
t=split(t,','); %divides each element of t at the delimiters specified by ','. The output t does not include the delimiters ','.(14807x8)
t=upper(t(:,:)); %converts all lowercase characters in str to the corresponding uppercase characters and leaves all other characters unchanged
uno=contains(t(:,1), '[01]'); %returns 1 (true) if the first column of t contains the specified pattern [01], and returns 0 (false) otherwise
due=contains(t(:,1), '[02]'); %returns 1 (true) if the first column of t contains the specified pattern [02], and returns 0 (false) otherwise
tre=contains(t(:,1), '[03]'); %returns 1 (true) if the first column of t contains the specified pattern [03], and returns 0 (false) otherwise
%creating an array of nth reception + information for 01
sens1=t(uno,4:8); %values only where 01 is present, from column 4 to 8 (five columns, the first one contains the nth reception from the 01 device, the other four the signal)
sens1=extractBetween(sens1,'[',']'); % extracts the substring from sens1 that occurs between the substrings [ and ]. The extracted substring does not include [ and ]. Elimination of [].
conc1=strcat(sens1(:,2),sens1(:,3), sens1(:,4), sens1(:,5)); %horizontally concatenates the columns from 2 to 5 of sens1 (last four columns of the original line, the ones that contain the information)
sens1=[sens1(:,1), conc1]; %sens1 is now composed of the the first column of sens1 (hexadecimal orderd numbers) and conc1 (nth reception + information)
%creating an array of nth reception + information for 02
sens2=t(due,4:8); %values only where 02 is present, from column 4 to 8 (five columns, the first one contains the nth reception from the 02 device, the other four the signal)
sens2=extractBetween(sens2,'[',']'); % extracts the substring from sens2 that occurs between the substrings [ and ]. The extracted substring does not include [ and ]. Elimination of [].
conc2=strcat(sens2(:,2),sens2(:,3), sens2(:,4), sens2(:,5)); %horizontally concatenates the columns from 2 to 5 of sens2 (last four columns of the original line, the ones that contain the information)
sens2=[sens2(:,1), conc2]; %sens2 is now composed of the the first column of sens2 (hexadecimal orderd numbers) and conc2 (nth reception + information)
%creating an array of nth reception + information for 03
sens3=t(tre,4:8); %values only where 03 is present, from column 4 to 8 (five columns, the first one contains the nth reception from the 03 device, the other four the signal)
sens3=extractBetween(sens3,'[',']'); % extracts the substring from sens3 that occurs between the substrings [ and ]. The extracted substring does not include [ and ]. Elimination of []
conc3=strcat(sens3(:,2),sens3(:,3), sens3(:,4), sens3(:,5)); %horizontally concatenates the columns from 2 to 5 of sens3 (last four columns of the original line, the ones that contain the information)
sens3=[sens3(:,1), conc3]; %sens2 is now composed of the the first column of sens3 (hexadecimal orderd numbers) and conc3 (nth reception + information)
% n1=sens1(:,1)
% n1=hex2dec(n1)

% max creation: identification of the longest among sens1, sens2, sens3; floor of (the longest)/256 
if (length(sens3)>length(sens1) && length(sens3)>length(sens2))
    max=floor((length(sens3))/256); %floor: rounds each element of X towrads minus inf
elseif (length(sens1)>length(sens2) && length(sens1)>length(sens3))
    max=floor((length(sens1))/256);
else
    max=floor((length(sens2))/256);
end
% conversion of the first column (nth reception) from hexadecimal to
% decimal
sens1(:,1)=hex2dec(sens1(:,1));
sens2(:,1)=hex2dec(sens2(:,1));
sens3(:,1)=hex2dec(sens3(:,1));
%01: since some receptions is lost, there is the need to assign the reception to the right number, inserting an empty line where the receprion is lost 
if (i_1==0)
    for j=1:1:255 %for s1 I take only the first 255 values of sense1
        if str2double(sens1(l_1,1))==j %if the l_1 value of the first column is equal to j (j starts from 1, as sens1 first column)
%         if n1(l)==j
            s1(j,:)=sens1(l_1,:);   %then assign the l_1 row to the j raw of s1
            l_1=l_1+1;              %update of l_1 means that I can go on with the row values
        else                        %if j is different from the l_1 value of the first column this means that the there is a reception "jump"
            s1(j,:)=["", ""];       %since there is a value lost, an empty row is added, no update of l_1 in order to reorder correctly
        end
    end
    i_1=i_1+1; %update of i_1 
    stato_1=1; %flag stato_1 up allows to go on with next if, after the termination of the procedure
end
%this part perform the same task from the 256th value of s1 till the end,
%taking the value from sens1 and leaving a blank cell when there is a
%"jump"
if (stato_1==1)
  while (i_1<(max+1)) %max created at lines 59-66. i_1*256 allows a division in blocks of 256, with values inside explored one by one by means of j
    for j=0:1:255
        if (var_1==0)
          if (l_1==length(sens1)+1)         %control: if l_1 is higher than the length of sens1 from now on put blank cells
              s1((j+i_1*256),:)=["", ""];
%               l=l+1;
               var_1=1;
          elseif str2double(sens1(l_1,1))==j %if there is correspondance between j and the value of the first column of sens1 at l_1 (no jump, missing value)
            s1((j+i_1*256),:)=sens1(l_1,:);  %then add the line of sens1 to s1
            l_1=l_1+1;                       %update l_1 to go to the next row
          else                               %if there is no correspondance means that there is a jump, a missing value
            s1((j+i_1*256),:)=["", ""];      %then put a blank cell to reorder
          end
        else                                %var_1=1 means that l_1 is higher than the length of sens1, so from now on put blank cells
           s1((j+i_1*256),:)=["", ""];
        end
    end
    i_1=i_1+1; %update i_1 to continue the while cycle
  end
end
%02: since some receptions is lost, there is the need to assign the reception to the right number, inserting an empty line where the receprion is lost
if (i_2==0)
    for j=1:1:255 %for s2 I take only the first 255 values of sense2
        if str2double(sens2(l_2,1))==j  %if the l_2 value of the first column is equal to j (j starts from 1, as sens2)
%         if n1(l)==j
            s2(j,:)=sens2(l_2,:);   %then assign the l_2 row to the j raw of s2
            l_2=l_2+1;              %update of l_2 means that I can go on with the row values
        else                        %if j is different from the l_2 value of the first column this means that the there is a reception "jump"
            s2(j,:)=["", ""];       %since there is a value lost, an empty row is added, no update of l_2 in order to reorder correctly
        end
    end
    i_2=i_2+1; %update of i_2 
    stato_2=1; %flag stato_2 up allows to go on with next if, after the termination of the procedure
end
%this part perform the same task from the 256th value of s1 till the end,
%taking the value from sens1 and leaving a blank cell when there is a
%"jump"
if (stato_2==1)
  while (i_2<(max+1))  %max created at lines 59-66. i_2*256 allows a division in blocks of 256, with values inside explored one by one by means of j
    for j=0:1:255
        if (var_2==0)
          if (l_2==length(sens2)+1) %control: if l_2 is higher than the length of sens2 from now on put blank cells
              s2((j+i_2*256),:)=["", ""];
%               l=l+1;
               var_2=1;
          elseif str2double(sens2(l_2,1))==j %if there is correspondance between j and the value of the first column of sens1 at l_2 (no jump, missing value)
            s2((j+i_2*256),:)=sens2(l_2,:);  %then add the line of sens2 to s2
            l_2=l_2+1;                       %update l_2 to go to the next row
          else                               %if there is no correspondance means that there is a jump, a missing value
            s2((j+i_2*256),:)=["", ""];      %then put a blank cell to reorder
          end
        else                                 %var_2=1 means that l_2 is higher than the length of sens2, so from now on put blank cells
           s2((j+i_2*256),:)=["", ""];       
        end
    end
    i_2=i_2+1; %update i_2 to continue the while cycle
  end
end
%03: since some receptions is lost, there is the need to assign the reception to the right number, inserting an empty line where the receprion is lost
if (i_3==0)
    for j=1:1:255  %for s3 I take only the first 255 values of sense3
        if str2double(sens3(l_3,1))==j %if the l_3 value of the first column is equal to j (j starts from 1, as sens3)
%         if n1(l)==j
            s3(j,:)=sens3(l_3,:);   %then assign the l_3 row to the j row of s3
            l_3=l_3+1;              %update of l_3 means that I can go on with the row values
        else                        %if j is different from the l_3 value of the first column this means that the there is a reception
            s3(j,:)=["", ""];       %since there is a value lost, an empty row is added, no update of l_3 in order to reorder correctly
        end
    end
    i_3=i_3+1; %update of i_3
    stato_3=1; %flag stato_3 up allows to go on with next if, after the termination of the procedure
end
%this part perform the same task from the 256th value of s1 till the end,
%taking the value from sens1 and leaving a blank cell when there is a
%"jump"
if (stato_3==1)
  while (i_3<(max+1)) %max created at lines 59-66. i_3*256 allows a division in blocks of 256, with values inside explored one by one by means of j
    for j=0:1:255
        if (var_3==0)
          if (l_3==length(sens3)+1) %control: if l_3 is higher than the length of sens3 from now on put blank cells
              s3((j+i_3*256),:)=["", ""];
%               l=l+1;
               var_3=1;
          elseif str2double(sens3(l_3,1))==j %if there is correspondance between j and the value of the first column of sens1 at l_3 (no jump, missing value)
            s3((j+i_3*256),:)=sens3(l_3,:);  %then add the line of sens3 to s3
            l_3=l_3+1;                       %update l_3 to go to the next row
          else                               %if there is not a correspondance means that there is a jump, a missing value
            s3((j+i_3*256),:)=["", ""];      %then put a blank cell to reorder
          end
        else        %var_3=1 means that l_3 is higher than the length of sens3, so from now on put blank cells
           s3((j+i_3*256),:)=["", ""];
        end
    end
    i_3=i_3+1; %update i_3 to continue the while cycle
  end
end

S1=s1(:,2); %S1 is only the second column of s1, only the information part
S2=s2(:,2); %S2 is only the second column of s2, only the information part
S3=s3(:,2); %S3 is only the second column of s3, only the information part

for i=1:1:length(S1) %creating the quaternion for the device 01
   
    temp=char(S1(i)); 
    if (~isempty(temp))                          %if the current i-th cell is not empty
    %if and(not(temp==' '),length(temp)==8)
    %if (not(temp==' ')& length(temp)==8 & strcmp(temp,'00000000')==0)
        quat1_1temp=hex2dec(cellstr(temp(1:2))); %cellstr: Convert to cell array of character vectors; taking the characters 1 and 2 of temp and converting from hexadecimal to decimal 
        if (quat1_1temp>127)
            quat1_1temp=quat1_1temp-256;         %put everything in the range [-128;127]
           
        end
        quat1_2temp=hex2dec(cellstr(temp(3:4))); %cellstr: Convert to cell array of character vectors; taking the characters 3 and 4 of temp and converting from hexadecimal to decimal
        if (quat1_2temp>127)
            quat1_2temp=quat1_2temp-256;
             
        end  
        quat1_3temp=hex2dec(cellstr(temp(5:6))); %cellstr: Convert to cell array of character vectors; taking the characters 5 and 6 of temp and converting from hexadecimal to decimal
        if (quat1_3temp>127)
           quat1_3temp=quat1_3temp-256;
            
        end  
        quat1_4temp=hex2dec(cellstr(temp(7:8))); %cellstr: Convert to cell array of character vectors; taking the characters 7 and 8 of temp and converting from hexadecimal to decimal
        if (quat1_4temp>127)
            quat1_4temp=quat1_4temp-256;
           
        end  
        quat1_1=[quat1_1; quat1_1temp/127];      %composing the quaternion normalized between [-1;1]
        quat1_2=[quat1_2; quat1_2temp/127];
        quat1_3=[quat1_3; quat1_3temp/127];
        quat1_4=[quat1_4; quat1_4temp/127];
    else                                         %if the current i-th cell is empty put NaN
        quat1_1=[quat1_1; NaN];
        quat1_2=[quat1_2; NaN];
        quat1_3=[quat1_3; NaN];
        quat1_4=[quat1_4; NaN];
    end
end

for i=1:1:length(S2) %creating the quaternion for the device 02
   
   temp=char(S2(i));
   %if and(not(temp==' '),length(temp)==8)
   if (~isempty(temp))                           %if the current i-th cell is not empty
   %if (not(temp==' ')& length(temp)==8 & strcmp(temp,'00000000')==0)
        quat2_1temp=hex2dec(cellstr(temp(1:2))); %cellstr: Convert to cell array of character vectors; taking the characters 1 and 2 of temp and converting from hexadecimal to decimal
        if (quat2_1temp>127)
            quat2_1temp=quat2_1temp-256;         %put everything in the range [-128;127]
           
        end
        quat2_2temp=hex2dec(cellstr(temp(3:4))); %cellstr: Convert to cell array of character vectors; taking the characters 3 and 4 of temp and converting from hexadecimal to decimal
        if (quat2_2temp>127)
        quat2_2temp=quat2_2temp-256;
           
        end  
        quat2_3temp=hex2dec(cellstr(temp(5:6))); %cellstr: Convert to cell array of character vectors; taking the characters 5 and 6 of temp and converting from hexadecimal to decimal
        if (quat2_3temp>127)
           quat2_3temp=quat2_3temp-256;
         
        end  
        quat2_4temp=hex2dec(cellstr(temp(7:8))); %cellstr: Convert to cell array of character vectors; taking the characters 7 and 8 of temp and converting from hexadecimal to decimal
        if (quat2_4temp>127)
            quat2_4temp=quat2_4temp-256;
            
        end  
       
        quat2_1=[quat2_1; quat2_1temp/127];      %composing the quaternion normalized between [-1;1]
        quat2_2=[quat2_2; quat2_2temp/127];
        quat2_3=[quat2_3; quat2_3temp/127];
        quat2_4=[quat2_4; quat2_4temp/127];
    
   else                                          %if the current i-th cell is empty put NaN
        quat2_1=[quat2_1; NaN];
        quat2_2=[quat2_2; NaN];
        quat2_3=[quat2_3; NaN];
        quat2_4=[quat2_4; NaN];
   end
end
for i=1:1:length(S3) %creating the quaternion for the device 03
   
    temp=char(S3(i));
    if (~isempty(temp))                          %if the current i-th cell is not empty
    %if and(not(temp==' '),length(temp)==8)
    %if (not(temp==' ')& length(temp)==8 & strcmp(temp,'00000000')==0)
        quat3_1temp=hex2dec(cellstr(temp(1:2))); %cellstr: Convert to cell array of character vectors; taking the characters 1 and 2 of temp and converting from hexadecimal to decimal
        if (quat3_1temp>127)
            quat3_1temp=quat3_1temp-256;         %put everything in the range [-128;127]
           
        end
        quat3_2temp=hex2dec(cellstr(temp(3:4))); %cellstr: Convert to cell array of character vectors; taking the characters 3 and 4 of temp and converting from hexadecimal to decimal
        if (quat3_2temp>127)
               quat3_2temp=quat3_2temp-256;
          
        end  
        quat3_3temp=hex2dec(cellstr(temp(5:6))); %cellstr: Convert to cell array of character vectors; taking the characters 5 and 6 of temp and converting from hexadecimal to decimal
        if (quat3_3temp>127)
                quat3_3temp=quat3_3temp-256;
         
        end  
        quat3_4temp=hex2dec(cellstr(temp(7:8))); %cellstr: Convert to cell array of character vectors; taking the characters 7 and 8 of temp and converting from hexadecimal to decimal
        if (quat3_4temp>127)
              quat3_4temp=quat3_4temp-256;
           
        end  
       
       
        quat3_1=[quat3_1; quat3_1temp/127];      %composing the quaternion normalized between [-1;1]
        quat3_2=[quat3_2; quat3_2temp/127];
        quat3_3=[quat3_3; quat3_3temp/127];
        quat3_4=[quat3_4; quat3_4temp/127];
    
    else                                         %if the current i-th cell is empty put NaN
        quat3_1=[quat3_1; NaN];
        quat3_2=[quat3_2; NaN];
        quat3_3=[quat3_3; NaN];
        quat3_4=[quat3_4; NaN];
    end
end


% Sensor 1
figure
plot(quat1_1); hold on; plot(quat1_2); hold on; plot(quat1_3); hold on; plot(quat1_4)
title ('Sensor 1');
legend ('q1', 'q2', 'q3', 'q4');
ylim([-1 1]);
% sensor 2
figure
plot(quat2_1); hold on; plot(quat2_2); hold on; plot(quat2_3); hold on; plot(quat2_4)
title ('Sensor 2');
legend ('q1', 'q2', 'q3', 'q4');
ylim([-1 1]);
% sensor 3
figure
plot(quat3_1); hold on; plot(quat3_2); hold on; plot(quat3_3); hold on; plot(quat3_4)
title ('Sensor 3');
legend ('q1', 'q2', 'q3', 'q4');
ylim([-1 1]);

%quat 1
figure
plot(quat1_1)
hold on
plot(quat2_1)
hold on
plot(quat3_1)
legend ('s1','s2','s3')
title('q1')
ylim([-1 1]);

%quat 2
figure
plot(quat1_2)
hold on
plot(quat2_2)
hold on
plot(quat3_2)
legend ('s1','s2','s3')
title('q2')
ylim([-1 1]);

%quat 3
figure
plot(quat1_3)
hold on
plot(quat2_3)
hold on
plot(quat3_3)
legend ('s1','s2','s3')
title('q3')
ylim([-1 1]);

%quat 4
figure
plot(quat1_4)
hold on
plot(quat2_4)
hold on
plot(quat3_4)
legend ('s1','s2','s3')
title('q4')
ylim([-1 1]);

add=[quat2_1 quat2_2 quat2_3 quat2_4]; %addominal quaternion (02)
tor=[quat1_1 quat1_2 quat1_3 quat1_4]; %thoracic quaternion (01)
ref=[quat3_1 quat3_2 quat3_3 quat3_4]; %reference quaternion (03)

figure
subplot(3,1,1) 
plot(tor) 
subplot(3,1,2)
plot(add)
%legend('prima','seconda','terza','quarta')
legend('first','second','third','forth')
subplot(3,1,3)
plot(ref)
%replacing NaN with interpolation in add, tor and ref
tor=tolgoNaN_c(tor);
add=tolgoNaN_c(add);
ref=tolgoNaN_c(ref);

%clearvars -except tor add ref 

%% Analysis PC_Ylenia
SgolayWindow=31; % grado smoothing (maggiore valore maggiore smoothing) da regolare per stima F
SgolayWindowPCA=31;
fDispo=10;
figure
% ref=movmean(ref,97,'includenan');
% abc=ref;
% ref=tor;
% tor=abc;
%add=tor;
%tor=add;
subplot(3,1,1) 
plot(tor)
% xlim([0 2000]) 
title('Select pose window') %seleziono la finestra da utilizzare come riferimento (posizione 0) 
subplot(3,1,2)
plot(add)
legend('prima','seconda','terza','quarta')
% xlim([0 2000])
subplot(3,1,3)
%plot(refnoNan(:,2))
% ref=movmean(ref,97,'includenan');
plot(ref)
% xlim([0 2000])

F=[];
i=1;

F(i,:)=round(ginput(1));i=i+1; %inizio sessione da tenere, ginput(n) allows you to identify the coordinates of n points.  
%To choose a point, move your cursor to the desired location and press either a mouse button or a key on the keyboard. 
%Press the Return key to stop before all n points are selected. MATLAB® returns the coordinates of your selected points.
pause
F(i,:)=round(ginput(1));i=i+1; %fine sessione da tenere
pause
%F(1,1)= 288; fixed values to compare with Python file
%F(2,1)= 4588;
tor_pose_w=mean(tor(F(1,1):F(2,1),:));% dati torace raw intervallo da tenere 
add_pose_w=mean(add(F(1,1):F(2,1),:)); %dati addome raw intervallo da tenere
ref_pose_w=mean(ref(F(1,1):F(2,1),:));% dati riferimento raw intervallo da tenere

Add_pose=[];
for i=1:length(add)                 
   Add_pose=[Add_pose; add_pose_w];  %values are all equal to add_pose_w (mean of the values between F(1,1) and F(2,1))
end
Ref_pose=[];
for i=1:length(ref)
   Ref_pose=[Ref_pose; ref_pose_w]; %values are all equal to ref_pose_w (mean of the values between F(1,1) and F(2,1))
end
Tor_pose=[];
for i=1:length(tor)
   Tor_pose=[Tor_pose; tor_pose_w]; %values are all equal to tor_pose_w (mean of the values between F(1,1) and F(2,1))
end
ref_pose=quatmultiply(ref,quatconj(Ref_pose)); %calcolo posizione ZERO del riferimento 
tor_pose=quatmultiply(tor,quatconj(Tor_pose)); %calcolo posizione ZERO del torace
add_pose=quatmultiply(add,quatconj(Add_pose)); %calcolo posizione ZERO dell'addome

figure
subplot(3,1,1) 
plot(tor)
title('select window to analyze') %seleziono finestra da analizzare (può anche contenere la pose)
% xlim([0 2000])
subplot(3,1,2)
plot(add)
% xlim([0 2000])
subplot(3,1,3)
%plot(refnoNan(:,2))
plot(ref)
% xlim([0 2000])

G=[];
i=1;

G(i,:)=round(ginput(1));i=i+1; %inizio sessione da tenere
pause
G(i,:)=round(ginput(1));i=i+1; %fine sessione da tenere
pause
%G(1,1)= 219;
%G(2,1)= 4666;
Add_Ok=add_pose(G(1,1):G(2,1),:); %dati addome raw intervallo da tenere
Tor_Ok=tor_pose(G(1,1):G(2,1),:);% dati torace raw intervallo da tenere 
%Tor_Ok=refnoNan(G(1,1):G(2,1),:);
Ref_Ok=ref_pose(G(1,1):G(2,1),:);% dati riferimento e raw intervallo da tenere


%calcolo compound quaternion : Addome rispetto a Ref e Torace rispetto a
%Ref
t1=quatmultiply(Tor_Ok,quatconj(Ref_Ok));
a1=quatmultiply(Add_Ok,quatconj(Ref_Ok));
% Detrend % interp_T=movmean(t1,37,'includenan'); %
interp_T=movmean(t1,97,'includenan'); %returns an array of local 97-point mean values, where each mean is calculated over a sliding window of length 97 across neighboring elements of A. 'includenan' includes all NaN values in the calculation
interp_A=movmean(a1,97,'includenan');
figure
plot(t1);hold on; plot(interp_T)
figure
plot(a1);hold on; plot(interp_A)
t1=t1-interp_T; %subtraction mean value interp_T obtained with movmean
a1=a1-interp_A; %subtraction mean value interp_A obtained with movmean

%% Analysis with PCA method 
%Calcolo PCA (addome)
[coeff_add,score,latent,tsquared,explainedA,mu] = pca(a1);
FuseA_1=a1*coeff_add(:,1);
FuseA_2=a1*coeff_add(:,2);
FuseA_3=a1*coeff_add(:,3);
FuseA_4=a1*coeff_add(:,4);

figure
plot(a1)
hold on
plot(FuseA_1, 'LineWidth',2);
hold on;
plot(FuseA_2, 'LineWidth',2);
hold on;
plot(FuseA_3, 'LineWidth',2);
hold on;
plot(FuseA_4, 'LineWidth',2);

title('Abdomen: PCA components  vs. raw quaternion components')
legend('q0', 'q1', 'q2','q3','PCA_1', 'PCA_2','PCA_3', 'PCA_4')

%calcolo PCA (torace)
[coeff_tor,score,latent,tsquared,explainedT,mu] = pca(t1);
FuseT_1=t1*coeff_tor(:,1);
FuseT_2=t1*coeff_tor(:,2);
FuseT_3=t1*coeff_tor(:,3);
FuseT_4=t1*coeff_tor(:,4);
figure
plot(t1)
hold on
plot(FuseT_1, 'LineWidth',2);
hold on;
plot(FuseT_2, 'LineWidth',2);
hold on;
plot(FuseT_3, 'LineWidth',2);
hold on;
plot(FuseT_4, 'LineWidth',2);

title('Thorax: PCA components  vs. raw quaternion components')
legend('q0', 'q1', 'q2','q3','PCA_1', 'PCA_2','PCA_3', 'PCA_4')

% Preliminary Estimation of breathing frequency

%using Abdomen PCA

figure
plot(FuseA_1) %uso prima componente PCA 

fStimVec=[];
EstimSmoothA=sgolayfilt(FuseA_1, 3,SgolayWindowPCA); % smoothing. Applies a Savitzky-Golay finite impulse response (FIR) smoothing filter of polynomial order 3 and frame length SgolayWindowPCA=31 to the data in vector FuseA_1
diff=max(EstimSmoothA)-min(EstimSmoothA); %signal amplitude evaluation in order to define a threshold for the maxima identification
thr=diff*5/100; %threshold 5% of the maximum variation in signal amplitude

hold on
plot(EstimSmoothA)


[M,I] = findpeaks(EstimSmoothA,'MinPeakDistance',6,'MinPeakProminence',thr); %signal peaks; returns a vector with the local maxima (M) of the input signal vector and indices at which the peaks occur. 
%Only those peaks that have a relative importance of at least 'MinPeakProminence'(Prominence: how much the peak stands out due to itsintrinsic height and its location relative to other peaks). 
%When you specify a value for 'MinPeakDistance', the algorithm chooses the tallest peak in the signal and ignores all peaks within 'MinPeakDistance' of it.
hold on
plot(I,M,'r*')
title('First abdomen PC, filtered version and peaks')
for i=1:length(M)-1
    intraPicco=(I(i+1)-I(i))/fDispo; %interpeak distance is the estimated time of the duration of every breath
    fStim=1/intraPicco; %the frequency is the inverse of the interpeak time breath by breath 
    fStimVec=[fStimVec fStim]; 
end
fStimMean=mean(fStimVec); % the estimated breathing frequency is the mean of the computed frequencies breath by breath
fStimstd=std(fStimVec); %estimated frequency standard deviation
lowThresoldA=max(0.05,(fStimMean-fStimstd));%computation of the low-frequency threshold from the estimated frequency, the spectrum peak will be identified starting from this frequency
[pxxA,fA] = pwelch(FuseA_1,300,50,512,10); %PCA_1 abdomen spectrum (fA is the nomralized frequency vector)

%using Thorax PCA
fStimVec=[];
EstimSmoothT=sgolayfilt(FuseT_1, 3,SgolayWindowPCA);
figure
plot(FuseT_1)

hold on
plot(EstimSmoothT)

diff=max(EstimSmoothT)-min(EstimSmoothT); %valuto ampiezza segnale per definire una soglia per il calcolo dei massimi
thr=diff*5/100; %soglia 5% della massima variazione in ampiezza del segnale
[M,I] = findpeaks(EstimSmoothT,'MinPeakDistance',6,'MinPeakProminence',thr);
hold on
plot(I,M,'r*')
title('First thoracic PC, filtered version and peaks')
for i=1:length(M)-1
    intraPicco=(I(i+1)-I(i))/fDispo;
    fStim=1/intraPicco;
    fStimVec=[fStimVec fStim];
end
fStimMeanT=mean(fStimVec);
fStimstdT=std(fStimVec);
[pxxT,fT] = pwelch(FuseT_1,300,50,512,10);% escludo SVC(1000:11500)
lowThresoldT=max(0.05,(fStimMeanT-fStimstdT));
fStimVec=[];

lowThresold=min([lowThresoldA lowThresoldT]);

% Detezione massimi e minimi Abdomen
Signal=-FuseA_1;
in = find(fA>lowThresold)-1;
fi=find(fA>2);
[M,I] = findpeaks(pxxA(in(1):fi(1)));
[bFMax, BFI]= max(M); %max value and index that corresponds to the maximum value of M
figure
plot(fA,pxxA)
xlabel('Frequency (Hz)')
ylabel('Magnitude')
hold on
plot(fA(I(BFI)+in(1)-1),bFMax, 'r*')
title('Abdomen Spectrum and maximum')

bFSpettro=fA(I(BFI)+in(1)-1); %fpeak is the value of fA at the max position
f1=max(0.05,bFSpettro-0.4);
f2=bFSpettro+0.4;
% f1=0.01;
% f2=0.3
%low pass filter
ft_pl = f2;
fDispo=10;
Wn_pl= ft_pl/(fDispo/2);
[b,a] = butter(1,Wn_pl,'low'); %returns the transfer function coefficients of an first-order lowpass digital Butterworth filter with normalized cutoff frequency Wn_pl
LowFilt=filtfilt(b,a,Signal);  %performs zero-phase digital filtering by processing the input data, Signal
figure
plot(Signal);hold on;
plot(LowFilt)

%high pass filter (elimino continua)
ft_ph=f1;
Wn_ph= ft_ph/(fDispo/2);
[b,a] = butter(1,Wn_ph,'high'); %returns the transfer function coefficients of an first-order highpass digital Butterworth filter with normalized cutoff frequency Wn_ph
HighFilt=filtfilt(b,a,LowFilt); %performs zero-phase digital filtering by processing the input data, LowFilt
hold on
plot(HighFilt)
hold on;
if bFSpettro*60<12
    perc=15;
    distance=35; %min peak distance di 35 frames corrisponde ad una freq respiratoria di 17 resp/min (siamo conservativi)
    SgolayWindow=15;
end
if (bFSpettro*60>12&&bFSpettro*60<20)
    perc=8;
    distance=20; %min peak distance di 20 frames corrisponde ad una freq respiratoria di 30 resp/min (siamo conservativi)
    SgolayWindow=11;
end
if (bFSpettro*60>20&&bFSpettro*60<40)
    perc=5;
    distance=9; %min peak distance di 12 frames corrisponde ad una freq respiratoria di 50 resp/min (siamo conservativi)
    SgolayWindow=9;
end
if (bFSpettro*60>40&&bFSpettro<59)
    perc=4;
    distance=7; %min peak distance di 8 frames corrisponde ad una freq respiratoria di 75 resp/min (siamo conservativi)
    SgolayWindow=7;
end
if bFSpettro*60>59
    perc=3;
    distance=3; %min peak distance di 5 frames corrisponde ad una freq respiratoria di 120 resp/min (siamo conservativi)
    SgolayWindow=5;
end
MinIdx=[];
Minima=[];

SmoothSmoothA=sgolayfilt(HighFilt, 3,SgolayWindow);
Diff=max(SmoothSmoothA)-min(SmoothSmoothA);
thr=Diff*perc/100;
plot(SmoothSmoothA)
legend('Raw','LowFilt', 'Highfilt','Sgolay')
title('Raw, Low-filtered, BP-filtered, and Savitzky-Golay filtered version (Abd)')
[Maxima,MaxIdx] = findpeaks(SmoothSmoothA,'MinPeakDistance',distance,'MinPeakProminence',thr);
DataInv = 1.01*max(SmoothSmoothA)-SmoothSmoothA; %data inversion, minima become maxima, detectable with findpeaks

MinFindIdx=[];
MinIdx=[];
M=[];
I=[];
Minima=[];
for i=1:length(Maxima)-1
    [M I]=max(DataInv(MaxIdx(i):MaxIdx(i+1)));
    Minimum=SmoothSmoothA(I+MaxIdx(i)-1);
    [minima,MinFindIdx] = findpeaks(DataInv(MaxIdx(i):MaxIdx(i+1)));
    MinFindIdx=MinFindIdx+MaxIdx(i)-1;
    SelectedMin=max(MinFindIdx);
    SelectedMinValue=SmoothSmoothA(SelectedMin);
    thr2=2*Diff/100;
    if (abs(SelectedMinValue-Minimum))<abs(thr2)
        MinIdx=[MinIdx SelectedMin];
        minima=SelectedMinValue;
        Minima=[Minima minima];
    else
        MinIdx=[MinIdx I+MaxIdx(i)];
        Minima=[Minima Minimum];
    end
%  figure
%  plot(HighFilt);hold on; plot(MinIdx, Minima, 'r*');
end
figure
plot(SmoothSmoothA)
hold on
plot(MaxIdx, Maxima, 'r*')
hold on 
plot(MinIdx, Minima, 'g*')
title('Abdomen (Max-Min)')

T=[];
Ti=[];
Te=[];
bF=[];
Ti_Te=[]; 
for i=1:1:length(MinIdx)
    te=(MinIdx(i)- MaxIdx(i))/fDispo;
    ti=(MaxIdx(i+1)-MinIdx(i))/fDispo;
    Ti=[Ti ti];
    Te=[Te te];
    ti_te=ti/te;
    Ti_Te=[Ti_Te ti/te]; 
end
for i=1:1:length(MinIdx)-1
    ttot=(MinIdx(i+1)-MinIdx(i))/fDispo;
    bf=1/ttot*60;
    T=[T; ttot];
    bF=[bF; bf];
end
%paramters' mean values
Tmean_PCA=mean(T);
Timean_PCA=mean(Ti);
Temean_PCA=mean(Te);
bFmean_PCA=mean(bF);
bFSpettro_PCA=bFSpettro*60;
Ti_Te_PCA_A=mean(Ti_Te);
duty_PCA_A=mean(Ti(1:end-1)./T');
%dev. standard
Tmean_sd=std(T);
Timean_sd=std(Ti);
Temean_sd=std(Te);
bFmean_sd=std(bF);
bFSpettro_sd=bFSpettro*60;
Ti_Te_sd_A=std(Ti_Te);
duty_sd_A=std(Ti(1:end-1)./T');
PCA_A=[bFmean_PCA Timean_PCA Temean_PCA Ti_Te_PCA_A duty_PCA_A];
SD_A=[bFmean_sd Timean_sd Temean_sd Ti_Te_sd_A duty_sd_A];

% Detezione massimi e minimi Thorax
SignalT=FuseT_1;
% Signal_used=-Signal_used;

figure
plot(fT,pxxT)
xlabel('Frequency (Hz)')
ylabel('Magnitude')
in = find(fT>lowThresold)-1;
fi=find(fT>2);

[M,I] = findpeaks(pxxT(in(1):fi(1)));
[bFMax, BFI]=max(M);
hold on
plot(fT(I(BFI)+in(1)-1),bFMax, 'r*')
title('Spectrum Thorax')

bFSpettroT=fT(I(BFI)+in(1)-1);
f1=max(0.05,bFSpettroT-0.4);
f2=bFSpettroT+0.4;
% f1=0.01;
% f2=0.3
%passa basso
ft_pl = f2;
fDispo=10;
Wn_pl= ft_pl/(fDispo/2);
[b,a] = butter(1,Wn_pl,'low');
LowFiltT=filtfilt(b,a,SignalT);
figure
plot(SignalT);hold on;
plot(LowFiltT)

%filtro passa alto (elimino continua)
ft_ph=f1;
Wn_ph= ft_ph/(fDispo/2);
[b,a] = butter(1,Wn_ph,'high');
HighFiltT=filtfilt(b,a,LowFiltT);
hold on
plot(HighFiltT)

if bFSpettroT*60<12
    perc=15;
    distance=35; %min peak distance di 35 frames corrisponde ad una freq respirtoria di 17 resp/min (siamo conservativi)
    SgolayWindow=15;
end
if (bFSpettroT*60>12&&bFSpettroT*60<20)  %min peak distance di 20 frames corrisponde ad una freq respirtoria di 30 resp/min (siamo conservativi)
    perc=8;
    distance=20;
    SgolayWindow=11;
end
if (bFSpettroT*60>20&&bFSpettroT*60<40)  %min peak distance di 12 frames corrisponde ad una freq respirtoria di 50 resp/min (siamo conservativi)
    perc=5;
    distance=9;
    SgolayWindow=9;
end

if (bFSpettroT*60>40&&bFSpettroT<59)
    perc=4;
    distance=7; %min peak distance di 8 frames corrisponde ad una freq respirtoria di 75 resp/min (siamo conservativi)
    SgolayWindow=7;
end
if (bFSpettroT*60>59)
    perc=3;
    distance=3; %min peak distance di 5 frames corrisponde ad una freq respirtoria di 120 resp/min (siamo conservativi)
    SgolayWindow=5;
end
MinIdx=[];
Minima=[];
SmoothSmoothT=sgolayfilt(HighFiltT, 3,SgolayWindow);
Diff=max(SmoothSmoothT)-min(SmoothSmoothT);
thr=Diff*perc/100;

plot(SmoothSmoothT)
legend('Raw','LowFilt', 'Highfilt','Sgolay')
title('Raw, Low-filtered, BP-filtered, and Savitzky-Golay filtered version (Th)')

[Maxima,MaxIdx] = findpeaks(SmoothSmoothT,'MinPeakDistance',distance,'MinPeakProminence',thr);
DataInvT = 1.01*max(SmoothSmoothT) -SmoothSmoothT;

MinFindIdx=[];
MinIdx=[];
M=[];
I=[];
Minima=[];
for i=1:length(Maxima)-1
    [M I]=max(DataInvT(MaxIdx(i):MaxIdx(i+1)));
    Minimum=SmoothSmoothT(I+MaxIdx(i)-1);
    [minima,MinFindIdx] = findpeaks(DataInvT(MaxIdx(i):MaxIdx(i+1)));
    MinFindIdx=MinFindIdx+MaxIdx(i)-1;
    SelectedMin=max(MinFindIdx);
    SelectedMinValue=SmoothSmoothT(SelectedMin);
    thr2=2*Diff/100;
    if (abs(SelectedMinValue-Minimum))<abs(thr2)
        MinIdx=[MinIdx SelectedMin];
        minima=SelectedMinValue;
        Minima=[Minima minima];
    else
        MinIdx=[MinIdx I+MaxIdx(i)];
        Minima=[Minima Minimum];
    end
%  figure
%  plot(HighFilt);hold on; plot(MinIdx, Minima, 'r*');
end
figure
plot(SmoothSmoothT)
hold on
plot(MaxIdx, Maxima, 'r*')
hold on 
plot(MinIdx, Minima, 'g*')
title('Thorax (Max-Min)')

Tt=[];
Tit=[];
Tet=[];
bFt=[];
Tit_Tet=[];

for i=1:1:length(MinIdx)
    tet=(MinIdx(i)- MaxIdx(i))/fDispo;
    tit=(MaxIdx(i+1)-MinIdx(i))/fDispo;
    Tit=[Tit tit];
    Tet=[Tet tet];
    tit_tet=tit/tet;
    Tit_Tet=[Tit_Tet tit/tet];
end
for i=1:1:length(MinIdx)-1
    ttott=(MinIdx(i+1)-MinIdx(i))/fDispo;
    bft=1/ttott*60;
    Tt=[Tt; ttott];
    bFt=[bFt; bft];
end
%paramters' mean values
Tmean_PCA_T=mean(Tt);
Timean_PCA_T=mean(Tit);
Temean_PCA_T=mean(Tet);
bFmean_PCA_T=mean(bFt);
bFSpettro_PCA_T=bFSpettroT*60;
Ti_Te_PCA_T=mean(Tit_Tet);
duty_PCA_T=mean(Tit(1:end-1)./Tt');
%dev. standard
Tmean_sd_T=std(Tt);
Timean_sd_T=std(Tit);
Temean_sd_T=std(Tet);
bFmean_sd_T=std(bFt);
bFSpettro_sd_T=bFSpettroT*60;
Ti_Te_sd_T=std(Tit_Tet);
duty_sd_T=std(Tit(1:end-1)./Tt');
SD_T=[bFmean_sd_T Timean_sd_T Temean_sd_T Ti_Te_sd_T duty_sd_T];
PCA_T=[bFmean_PCA_T Timean_PCA_T Temean_PCA_T Ti_Te_PCA_T duty_PCA_T]; 

%PCA total
SmoothSmoothTot=SmoothSmoothT+SmoothSmoothA;
[pxxTot,fTot] = pwelch(SmoothSmoothTot,300,50,512,10); %calcolo lo spettro totale
in = find(fTot>lowThresold)-1;
fi=find(fTot>2);
[M,I] = findpeaks(pxxTot(in(1):fi(1)));
[bFMax, BFI]=max(M);
figure
plot(fTot,pxxTot)
xlabel('Frequency (Hz)')
ylabel('Magnitude')
hold on
plot(fTot(I(BFI)+in(1)-1),bFMax, 'r*')
title('Spectrum Total')

bFSpettroTot=fTot(I(BFI)+in(1)-1);
if bFSpettroTot*60<12
    perc=15;
    distance=35; %min peak distance di 35 frames corrisponde ad una freq respirtoria di 17 resp/min (siamo conservativi)
    SgolayWindow=15;
end
if (bFSpettroTot*60>12 && bFSpettroTot*60<20)  %min peak distance di 20 frames corrisponde ad una freq respirtoria di 30 resp/min (siamo conservativi)
    perc=8;
    distance=20;
    SgolayWindow=11;
end
if (bFSpettroTot*60>20 && bFSpettroTot*60<40)  %min peak distance di 12 frames corrisponde ad una freq respirtoria di 50 resp/min (siamo conservativi)
    perc=5;
    distance=9;
    SgolayWindow=9;
end

if (bFSpettroTot*60>40 && bFSpettroTot<59)
    perc=4;
    distance=7; %min peak distance di 8 frames corrisponde ad una freq respirtoria di 75 resp/min (siamo conservativi)
    SgolayWindow=7;
end
if (bFSpettroTot*60>59)
    perc=3;
    distance=3; %min peak distance di 5 frames corrisponde ad una freq respirtoria di 120 resp/min (siamo conservativi)
    SgolayWindow=5;
end
Diff=max(SmoothSmoothTot)-min(SmoothSmoothTot);
thr=Diff*perc/100;
[Maxima,MaxIdx] = findpeaks(SmoothSmoothTot,'MinPeakDistance',distance,'MinPeakProminence',thr);
DataInvTot = 1.01*max(SmoothSmoothTot)-SmoothSmoothTot;

MinFindIdx=[];
MinIdx=[];
M=[];
I=[];
Minima=[];
for i=1:length(Maxima)-1
[M I]=max(DataInvTot(MaxIdx(i):MaxIdx(i+1)));
Minimum=SmoothSmoothTot(I+MaxIdx(i)-1);
[minima,MinFindIdx] = findpeaks(DataInvTot(MaxIdx(i):MaxIdx(i+1)));
MinFindIdx=MinFindIdx+MaxIdx(i)-1;
SelectedMin=max(MinFindIdx);
SelectedMinValue=SmoothSmoothTot(SelectedMin);
thr2=2*Diff/100;
if (abs(SelectedMinValue-Minimum))<abs(thr2)
    MinIdx=[MinIdx SelectedMin];
    minima=SelectedMinValue;
    Minima=[Minima minima];
else
   MinIdx=[MinIdx I+MaxIdx(i)];
   Minima=[Minima Minimum];
end
%  figure
%  plot(HighFilt);hold on; plot(MinIdx, Minima, 'r*');
end
figure
plot(SmoothSmoothTot)
hold on
plot(MaxIdx, Maxima, 'r*')
hold on 
plot(MinIdx, Minima, 'g*')
title('Total')

Ttot=[];
Titot=[];
Tetot=[];
bFtot=[];
Tit_Tetot=[];


for i=1:1:length(MinIdx)
    tetot=(MinIdx(i)- MaxIdx(i))/fDispo;
    titot=(MaxIdx(i+1)-MinIdx(i))/fDispo;
    Titot=[Titot titot];
    Tetot=[Tetot tetot];
    tit_tet=titot/tetot;
    Tit_Tetot=[Tit_Tetot titot/tetot];
end
for i=1:1:length(MinIdx)-1
    ttott_tot=(MinIdx(i+1)-MinIdx(i))/fDispo;
    bftot=1/ttott_tot*60;
    Ttot=[Ttot; ttott_tot];
    bFtot=[bFtot; bftot];
end
Tmean_PCA_Tot=mean(Ttot);
Timean_PCA_Tot=mean(Titot);
Temean_PCA_Tot=mean(Tetot);
bFmean_PCA_Tot=mean(bFtot);

Ti_Te_PCA_Tot=mean(Tit_Tetot);
duty_PCA_Tot=mean(Titot(1:end-1)./Ttot');

%dev. standard
Tmean_sd_Tot=std(Ttot);
Timean_sd_Tot=std(Titot);
Temean_sd_Tot=std(Tetot);
bFmean_sd_Tot=std(bFtot);
Ti_Te_sd_Tot=std(Tit_Tetot);
duty_sd_Tot=std(Titot(1:end-1)./Ttot');
SD_TOT=[bFmean_sd_Tot Timean_sd_Tot Temean_sd_Tot Ti_Te_sd_Tot duty_sd_Tot];
PCA_tot=[bFmean_PCA_Tot Timean_PCA_Tot Temean_PCA_Tot Ti_Te_PCA_Tot duty_PCA_Tot]; 
%% Results
% Selected window
% Wind=[G(1,1) G(2,1)]
% 
%analysis on PCA_1 abdomen 
PCA_A
% Ti_Te_PCA_A
% duty_PCA_A
%analysis on PCA_1 thorax
PCA_T
% Ti_Te_PCA_T
% duty_PCA_T
%analysis on PCA_1 total
PCA_tot

SD_A
SD_T
SD_TOT