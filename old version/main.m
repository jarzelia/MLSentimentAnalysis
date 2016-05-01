% Main file for Sentiment Analysis %

% Eric Wang, eric_wang1@student.uml.edu
% Patrick Lehane, patrick_lehane@student.uml.edu

% CURRENT CODE ASSUMES YOU HAVE TRAINING DATA IN SAME DIRECTORY AS MAIN

dataPos = {}; dataNeg = {}; 
wordCount = {0,0,0};   %data sctructures

trainPos = dir('train/pos');   %creates structure of files in pos
trainNeg = dir('train/neg');   %   same for neg

%loads all reviews into two separate cell array
for i = 3:size(trainPos)(1,1);
  fileID = fopen(strcat('train/pos/', trainPos(i).name));
  dataPos((i - 2),1) = fgetl(fileID);
  fclose(fileID);
end

for i = 3:size(trainNeg)(1,1);
  fileID = fopen(strcat('train/neg/', trainNeg(i).name));
  dataNeg((i - 2),1) = fgetl(fileID);
  fclose(fileID);
end

%This block is really ugly
%nested for loops and bad lookups
%This parses the string word by word and increments the counter based on pos/neg
for i = 1:size(dataPos)(1,1)
  [token,remain] = strtok(dataPos(i,1));
  
  temp = true;
  while temp
    temp = !(isempty(remain));
    boolIndex = strcmp(token,wordCount);    %location of parsed word
    intIndex = find(boolIndex);
    if isempty(intIndex)                    %word not found
      wordCount = [wordCount;{token,1,0}];
    else                                    %word found
      wordCount{intIndex,2}++;
    end
    [token,remain] = strtok(remain);
  end
end

% Same, but for negative reviews
for i = 1:size(dataNeg)(1,1)
  [token,remain] = strtok(dataPos(i,1));
  
  temp = true;
  while temp
    temp = !(isempty(remain));
    boolIndex = strcmp(token,wordCount);    %location of parsed word
    intIndex = find(boolIndex);
    if isempty(intIndex)                    %word not found
      wordCount = [wordCount;{token,0,1}];
    else                                    %word found
      wordCount{intIndex,3}++;
    end
    [token,remain] = strtok(remain);
  end
end


%writes for export.  Hopefully this only needs ot be run once.
fileID = fopen('celldata.dat','w');
formatSpec = '%s %d %d %s\n';
numRows = size(wordCount)(1,1);
for i = 1:numRows
  fprintf(fileID,formatSpec,wordCount{i,:});
end 

fclose(fileID);



%%  temp = true;
%%  while temp
%%    for j = 1:size(wordCount)(1,1)
%%      if strcmp(wordCount{j,1},token)
%%        wordCount{j,2}++;
%%        break;
%%      end
%%    end
%%    X = size(wordCount)(1,1);
%%    wordCount
    
    
    
%keep in mind strtok to parse
