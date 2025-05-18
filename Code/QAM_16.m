clear all;
close all;

% Parameters
f = 1000;        % Frequency of the sinusoidal signal (Hz)
Fs = 4000;       % Sampling frequency (samples per second)
t = (1/Fs:1/Fs:1)';  % Time vector for 1 second (4000 samples)
Am = 1.0;        % Amplitude of the sinusoid

% Generate signal
signal = Am * sin(2 * pi * f * t);

% Plot original signal
figure(1);
plot(t(1:200), signal(1:200));
set(gca, 'ytick', [-1 0 1]);
title('A segment of synthetically generated sinusoidal waveform');
xlabel('time (sec)');
ylabel('Amplitude (volt)');
grid on;

% Quantization parameters
maximumvalue = max(signal);
minimumvalue = min(signal);
n_bits = 8;      % 8 bits per sample
n_levels = 2^n_bits;  % 256 levels
interval = (maximumvalue - minimumvalue) / (n_levels - 1);
partition = minimumvalue:interval:maximumvalue;
codebook = (minimumvalue - interval/2):interval:(maximumvalue + interval/2); % Centered codebook

% Manual quantization (replacing quantiz)
index = zeros(size(signal));
for i = 1:length(signal)
    [~, idx] = min(abs(codebook - signal(i)));
    index(i) = idx - 1;  % Zero-based indexing (0 to 255)
end

% Convert decimal indices to binary (replacing bitget)
num_samples = length(signal); % 4000
matrix = zeros(num_samples, n_bits);
for i = 1:num_samples
    bin_str = dec2bin(index(i), n_bits);
    matrix(i, :) = double(bin_str) - '0'; % Convert to binary array
end

% Prepare baseband bitstream
matrixtps = matrix';  % 8 rows x 4000 columns
baseband = reshape(matrixtps, [], 1);  % 32000 bits

% Bit rate parameters
Tb = 1 / (Fs * n_bits);  % Bit duration (1/32000 s)
time = 0:Tb:1;  % Time vector for bits

% Plot baseband signal
figure(2);
stairs(time(1:500), baseband(1:500));
title('A segment of baseband signal');
xlabel('Time (sec)');
ylabel('Binary value');
set(gca, 'ytick', [0 1]);
axis([0 time(500) 0 1]);
grid on;

% Convolutional Encoding (manual, replacing poly2trellis and convenc)
% Using a (2,1,7) convolutional code with generators [171, 133] (octal)
% Generator polynomials: g1 = 1+x+x^2+x^3+x^6, g2 = 1+x^2+x^3+x^5+x^6
input_to_Convolutional_encoder = baseband'; % 1 x 32000
m = 6; % Constraint length - 1
code = zeros(1, 2 * length(input_to_Convolutional_encoder));
state = zeros(1, m); % Shift register
g1 = [1 1 1 1 0 0 1]; % Generator 171 octal
g2 = [1 0 1 1 0 1 1]; % Generator 133 octal
for i = 1:length(input_to_Convolutional_encoder)
    input_bit = baseband(i);
    % Update shift register
    state = [input_bit, state(1:end-1)];
    % Output bits
    out1 = mod(sum(state .* g1(2:end)) + input_bit * g1(1), 2);
    out2 = mod(sum(state .* g2(2:end)) + input_bit * g2(1), 2);
    code(2*i-1:2*i) = [out1, out2];
end

% Interleaving (manual, replacing randintrlv)
% Use a block interleaver (80x800 matrix)
interleaver_rows = 80;
interleaver_cols = length(code) / interleaver_rows; % 800
if mod(length(code), interleaver_rows) ~= 0
    code = [code, zeros(1, interleaver_rows - mod(length(code), interleaver_rows))];
    interleaver_cols = length(code) / interleaver_rows;
end
code_matrix = reshape(code, interleaver_rows, interleaver_cols);
data_interleave = reshape(code_matrix', 1, []); % Read column-wise

% 16-QAM Modulation (replacing qammod)
M = 16;  % 16-QAM
k = log2(M);  % 4 bits per symbol
% Ensure data length is divisible by k
if mod(length(data_interleave), k) ~= 0
    data_interleave = [data_interleave, zeros(1, k - mod(length(data_interleave), k))];
end
% Manual bit-to-symbol (replacing bi2de)
num_symbols = floor(length(data_interleave) / k);
symbol = zeros(num_symbols, 1);
for i = 1:num_symbols
    bit_quad = data_interleave((i-1)*k + 1:i*k);
    symbol(i) = bit_quad(1)*8 + bit_quad(2)*4 + bit_quad(3)*2 + bit_quad(4); % 0000->0, ..., 1111->15
end
% 16-QAM constellation (standard square lattice, normalized)
constellation = [-3-3j, -3-1j, -3+1j, -3+3j, ...
                 -1-3j, -1-1j, -1+1j, -1+3j, ...
                  1-3j,  1-1j,  1+1j,  1+3j, ...
                  3-3j,  3-1j,  3+1j,  3+3j] / sqrt(10); % Normalize by sqrt(10)
Quadrature_amplitude_modulated_data = zeros(num_symbols, 1);
for i = 1:num_symbols
    Quadrature_amplitude_modulated_data(i) = constellation(symbol(i) + 1);
end

% Demodulation (manual, replacing qamdemod)
Quadrature_amplitude_demodulated_data = zeros(num_symbols, 1);
for i = 1:num_symbols
    [~, idx] = min(abs(Quadrature_amplitude_modulated_data(i) - constellation));
    Quadrature_amplitude_demodulated_data(i) = idx - 1; % 0-based symbols
end

% Symbol Error Rate (manual, replacing symerr)
symbol_errors = sum(symbol ~= Quadrature_amplitude_demodulated_data);
symbol_error_rate = symbol_errors / num_symbols;
fprintf('Symbol Error Rate: %.5f\n', symbol_error_rate);

% Symbol to bit conversion (manual, replacing de2bi)
Retrieved_bit = zeros(num_symbols * k, 1);
for i = 1:num_symbols
    sym = Quadrature_amplitude_demodulated_data(i);
    % Convert symbol to 4 bits
    bits = zeros(4, 1);
    bits(1) = bitget(uint8(sym), 4); % MSB
    bits(2) = bitget(uint8(sym), 3);
    bits(3) = bitget(uint8(sym), 2);
    bits(4) = bitget(uint8(sym), 1); % LSB
    Retrieved_bit((i-1)*k + 1:i*k) = bits;
end

% Deinterleaving (manual, replacing randdeintrlv)
deinterleave_matrix = reshape(Retrieved_bit, interleaver_cols, interleaver_rows)';
data_deinterleave = reshape(deinterleave_matrix, 1, []); % Read row-wise

% Convolutional Decoding (manual Viterbi, replacing vitdec)
decoded_bits = zeros(1, floor(length(data_deinterleave)/2));
state = zeros(1, m);
for i = 1:length(decoded_bits)
    r = data_deinterleave(2*i-1:2*i);
    min_dist = Inf;
    best_bit = 0;
    best_state = state;
    for input = 0:1
        temp_state = [input, state(1:end-1)];
        out1 = mod(sum(temp_state .* g1(2:end)) + input * g1(1), 2);
        out2 = mod(sum(temp_state .* g2(2:end)) + input * g2(1), 2);
        dist = sum(abs([out1, out2] - r));
        if dist < min_dist
            min_dist = dist;
            best_bit = input;
            best_state = temp_state;
        end
    end
    decoded_bits(i) = best_bit;
    state = best_state;
end
decod2 = decoded_bits';

% Bit Error Rate (manual, replacing biterr)
baseband = double(baseband(1:length(decod2)));
[number, bit_error_rate] = deal(sum(decod2 ~= baseband), sum(decod2 ~= baseband) / length(decod2));
fprintf('Bit Error Rate: %.5f\n', bit_error_rate);

% Reshape and compare with matrixtps
convert = reshape(decod2, n_bits, [])'; % 4000 x 8
matrixtps = double(matrixtps(:, 1:size(convert, 1))');
[number_matrix, matrix_error_rate] = deal(sum(sum(convert ~= matrixtps)), sum(sum(convert ~= matrixtps)) / numel(matrixtps));
fprintf('Matrix Bit Error Rate: %.5f\n', matrix_error_rate);

% Binary to decimal (manual, replacing bi2de)
intconv = zeros(size(convert, 1), 1);
for i = 1:size(convert, 1)
    intconv(i) = bin2dec(char(convert(i, :) + '0'));
end

% Decimal Error Rate (manual, replacing biterr)
[number_int, decimal_error_rate] = deal(sum(intconv ~= index(1:length(intconv))), sum(intconv ~= index(1:length(intconv))) / length(intconv));
fprintf('Decimal Error Rate: %.5f\n', decimal_error_rate);

% Reconstruct signal
sample_value = minimumvalue + intconv * interval;

% Plot original vs reconstructed signal
figure(3);
subplot(2,1,1);
plot(t(1:100), signal(1:100)); % Use t instead of time
set(gca, 'ytick', [-1 0 1]);
axis([0 t(100) -1 1]);
title('Graph for a segment of recorded Audio signal');
xlabel('Time (sec)');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
plot(t(1:min(100, length(sample_value))), sample_value(1:min(100, length(sample_value))));
set(gca, 'ytick', [-1 0 1]);
axis([0 t(100) -1 1]);
title('Graph for a segment of retrieved Audio signal');
xlabel('Time (sec)');
ylabel('Amplitude');
grid on;