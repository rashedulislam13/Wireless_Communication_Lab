% MATLAB program to evaluate performance of a 1/2-rate convolutionally encoded
% DS-CDMA system for a single user in Rayleigh fading with AWGN (no Communications Toolbox)

clear all;
close all;
clc;

% System Parameters
numBits = 1000;            % Number of message bits
bitRate = 1000;            % Bit rate in Hz (1 kHz)
tb = 1/bitRate;            % Bit duration in seconds
chipRate = 10000;          % Chip rate in Hz (10x bit rate)
tc = 1/chipRate;           % Chip duration in seconds
spreadingFactor = chipRate/bitRate; % Spreading factor (10 chips per bit)
eb = 0.5;                  % Energy per bit
fc = 5000;                 % Carrier frequency in Hz (5 kHz)
snr_dB = 0:1:10;           % SNR range in dB
samplesPerChip = 10;       % Samples per chip for signal generation
samplesPerBit = samplesPerChip * spreadingFactor; % Samples per encoded bit

% Generate random message bits
msg = randi([0 1], 1, numBits);

% Custom Convolutional Encoder (1/2-rate, generators [110, 111])
% Generator polynomials: g1 = [1 1 0], g2 = [1 1 1]
g1 = [1 1 0]; % First generator polynomial
g2 = [1 1 1]; % Second generator polynomial
mem = 2;      % Memory of the encoder (constraint length - 1)
state = [0 0]; % Initial shift register state (memory only)
encoded = zeros(1, 2 * numBits); % Output: 2 bits per input bit
for i = 1:numBits
    % Form full state: current input bit + memory
    full_state = [msg(i) state]; % 1x3 vector: [current_bit, memory1, memory2]
    % Compute outputs: XOR operations based on generator polynomials
    out1 = mod(sum(full_state .* g1), 2); % First output bit
    out2 = mod(sum(full_state .* g2), 2); % Second output bit
    encoded(2*i-1:2*i) = [out1 out2];
    % Update memory: shift in the current bit
    state = [msg(i) state(1:end-1)];
end

% BPSK Modulation: Map 0 to -1, 1 to +1
modulated = 2 * encoded - 1; % Bipolar NRZ format

% Generate PN sequence for spreading
seed = [1 -1 1 -1]; % Initial seed in bipolar NRZ format
pn = zeros(1, length(modulated) * spreadingFactor);
seed_temp = seed;
for i = 1:length(modulated) * spreadingFactor
    pn(i) = seed_temp(4);
    % Generate next bit using XOR of positions 3 and 4
    if seed_temp(4) == seed_temp(3)
        temp = -1;
    else
        temp = 1;
    end
    % Shift register
    seed_temp(4) = seed_temp(3);
    seed_temp(3) = seed_temp(2);
    seed_temp(2) = seed_temp(1);
    seed_temp(1) = temp;
end

% Upsample modulated signal to chip rate
t = (tc/samplesPerChip):(tc/samplesPerChip):tb*length(modulated); % Time vector
baseband = zeros(1, length(t));
for i = 1:length(modulated)
    baseband((i-1)*samplesPerBit+1:i*samplesPerBit) = modulated(i);
end

% Plot a segment of the baseband signal
figure(1);
stairs(t(1:800), baseband(1:800));
xlabel('Time (sec)');
ylabel('Binary Value');
set(gca, 'ytick', [-1 1]);
title('Segment of Baseband Signal for Single User');

% BPSK Modulation
bpskmod = sqrt(2*eb) * baseband .* cos(2*pi*fc*t);

% Spread the signal
pn_upsampled = zeros(1, length(t));
for i = 1:length(pn)
    pn_upsampled((i-1)*samplesPerChip+1:i*samplesPerChip) = pn(i);
end
sigtx = bpskmod .* pn_upsampled;

% Plot a segment of the transmitted signal
figure(2);
plot(t(1:200), sigtx(1:200));
xlabel('Time (sec)');
ylabel('Amplitude');
title('Segment of Transmitted DS-CDMA Signal');
grid on;

% Frequency Domain Analysis
spectrum = abs(fft(bpskmod));
number = length(t);
sampling_frequency = 1/(tc/samplesPerChip);
frequency = (0:number-1)*(sampling_frequency/number);
figure(3);
plot(frequency(1:floor(number/2)), spectrum(1:floor(number/2)));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('Frequency Domain Analysis of BPSK Modulated Signal');
grid on;

% Initialize BER array
ber = zeros(1, length(snr_dB));

% Simulation Loop over SNR
for m = 1:length(snr_dB)
    % Rayleigh Fading (slow fading, constant over each bit)
    % Generate complex Gaussian fading (Rayleigh magnitude)
    fading = (randn(1, length(modulated)) + 1i*randn(1, length(modulated))) / sqrt(2);
    fading = abs(fading); % Rayleigh fading envelope
    fadedsig = zeros(1, length(t));
    for i = 1:length(modulated)
        fadedsig((i-1)*samplesPerBit+1:i*samplesPerBit) = sigtx((i-1)*samplesPerBit+1:i*samplesPerBit) * fading(i);
    end
    
    % Add AWGN
    % Convert SNR from dB to linear, adjust for code rate (1/2)
    snr_linear = 10^(snr_dB(m)/10);
    sigma = sqrt(eb/(2 * snr_linear)); % Noise standard deviation
    noise = sigma * randn(1, length(t));
    composite_signal = fadedsig + noise;
    
    % Despreading
    rx = composite_signal .* pn_upsampled;
    
    % BPSK Demodulation
    demodcar = sqrt(2*eb) * cos(2*pi*fc*t);
    bpskdemod = rx .* demodcar;
    
    % Integrate over each bit period
    sum_out = zeros(1, length(modulated));
    for i = 1:length(modulated)
        sum_out(i) = sum(bpskdemod((i-1)*samplesPerBit+1:i*samplesPerBit));
    end
    
    % Hard decision
    rxbits = sum_out > 0;
    
    % Simplified Decoding (approximate inversion of encoding)
    % Note: This is a basic approach; Viterbi would be more accurate
    decoded = zeros(1, numBits);
    state = [0 0];
    for i = 1:numBits
        % Simulate decoding by checking pairs of received bits
        rx_pair = rxbits(2*i-1:2*i);
        % Estimate input bit based on majority logic (simplified)
        full_state = [0 state]; % Assume unknown input, use memory
        out1 = mod(sum(full_state .* g1), 2);
        out2 = mod(sum(full_state .* g2), 2);
        % Compare received pair to expected outputs
        if sum(abs(rx_pair - [out1 out2])) <= 1
            decoded(i) = 0;
        else
            decoded(i) = 1;
        end
        % Update state
        state = [decoded(i) state(1)];
    end
    
    % Compute BER
    ber(m) = sum(abs(decoded - msg)) / numBits;
end             

% Plot BER
figure(4);
semilogy(snr_dB, ber, 'b-o', 'LineWidth', 2);
xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER for 1/2-Rate Coded DS-CDMA in Rayleigh Fading with AWGN');
legend('Single User (Simplified Decoding)');
grid on;