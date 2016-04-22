
%% Initialization
clear ; close all; clc

% Load training data (created by 'parser.py')
path_theData = 'dataio/theData.dat';

if(exist(path_theData))
   [test_class, test_tf, test_words] = textread(path_theData, '%d %d %s');
else
	% Error: DNE
   disp(['Could not open ' path_theData '. File does not exist.']);
end

% Some random test
thisFile = fopen('test/0.txt', 'r');
thatFile = fopen('test/1.txt', 'r');

% Removing nonalphanumerics
pattern = '[^a-zA-Z\d\s:]';
A = fileread('test/1.txt');
A = lower(A);
A = regexprep(A, pattern, '');

tokens = strsplit(A);
tokens_size = size(tokens)(1,2);


% m - number of examples
% n - number of features

X = zeros(10999, 1000);
for j = 1:tokens_size
	idx = find ( strcmp(test_words, tokens{j}) );

	if !isempty(idx)			% Word Exists in Training Set
		disp('Found!');

		if size(idx)(1,1) == 2
			X(1, idx(1)) += 1;
		end

	else
		disp('Not Found!');
	end
end

%C = textscan(thisFile, '%s');
%C = textscan(thatFile, '%s');

%A = fscanf(thatFile, '%s');



%someLine = fgetl(someFile);
%while ischar(someLine)
%	disp(someLine)
%	someLine = fgetl(someFile);
%end
%disp(regexp(test_words, 'good', 'match'));

fclose(thisFile);
fclose(thatFile);