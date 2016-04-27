

fileIDp = fopen('dataio/pos_datae.dat');
fileIDn = fopen('dataio/neg_datae.dat');

formatSpec = '%d %s';

A = textscan(fileIDp,formatSpec);
fclose(fileIDp);

B = textscan(fileIDn,formatSpec);
fclose(fileIDn);
C = {};

for i = 1:size(A{1,1})(1,1)
  boolIndex = strcmp(A{1,2}{i,1},B{1,2});
  intIndex = find(boolIndex);
  temp = A{1,2}{i,1};
  if isempty(intIndex)
    C = [C;{1,temp}];
  elseif ((0.75 * A{1,1}(i)) > B{1,1}(intIndex))
    C = [C;{1,temp}];
  end
end

for i = 1:size(B{1,1})(1,1)
  boolIndex = strcmp(B{1,2}{i,1},A{1,2});
  intIndex = find(boolIndex);
  temp = B{1,2}{i,1};
  if isempty(intIndex)
    C = [C;{0,temp}];
  elseif ((0.75 * B{1,1}(i)) > A{1,1}(intIndex))
    C = [C;{0,temp}];
  end
end

fileID = fopen('celldata.dat','w');
formatSpec = '%d %s \n';
for i = 1:size(C)(1,1)
  fprintf(fileID,formatSpec,C{i,:});
end 

fclose(fileID);