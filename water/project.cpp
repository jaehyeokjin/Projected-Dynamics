/***********************************************************************
 * Projected Force Calculation using FFTW3, OpenMP and MPI
 *
 * This code:
 *  (1) Reads a kernel file "K.out" (time, kernel, junk) sampled every 2 dt.
 *      It builds a cubic-spline interpolation for t < 600 and fits an
 *      exponential decay (via linear regression on log(K)) over t = 600–2500,
 *      then replaces the kernel for t >= 600 up to TOTAL_TIME.
 *      The final kernel is written to "K_extrapolated.txt".
 *
 *  (2) Reads a LAMMPS trajectory file ("lammpstrj") which (ignoring header
 *      quirks) contains: id, type, x, y, z, vx, vy, vz, fx, fy, fz.
 *
 *  (3) For each particle (mass = 18.015) and for each time frame,
 *      it computes:
 *
 *         Project_I(t) = F_I(t) + factor * ∫₀ᵗ K(t-s) * [m * V_I(s)] ds,
 *
 *      where factor = 1e4/4.184. The convolution is computed via FFTW3.
 *
 *  (4) The projected force is appended to the original particle data and
 *      written out as "projected_lammpstrj.txt".  MPI and OpenMP are used
 *      to speed up the convolution.
 *
 * Compile with (example):
 *    mpicxx -fopenmp -O3 projected_forces.cpp -lfftw3 -lfftw3_omp -lm -o projected_forces
 *
 ***********************************************************************/

#include <mpi.h>
#include <omp.h>
#include <fftw3.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;

// Global constants
const double mass = 18.015;
const double factor = 1e4 / 4.184;
const int TOTAL_TIME = 200000;  // Total dt for the kernel extrapolation

// --- Structures for trajectory data ---
struct Particle {
    int id;
    int type;
    double x, y, z;
    double vx, vy, vz;
    double fx, fy, fz;
};

struct Frame {
    int timestep;
    vector<Particle> particles; // size = num_particles
};

// --- Cubic spline functions ---
// Given arrays x and y (size n) and vector y2 (to hold 2nd derivatives),
// this routine computes the second derivatives for a natural cubic spline.
void compute_spline(const vector<double>& x, const vector<double>& y, vector<double>& y2) {
    int n = x.size();
    vector<double> u(n, 0.0);
    y2[0] = 0.0;
    u[0] = 0.0;
    for (int i = 1; i < n - 1; i++) {
        double sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        double p = sig * y2[i - 1] + 2.0;
        y2[i] = (sig - 1.0) / p;
        double dd = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
        u[i] = (6.0 * dd / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
    }
    y2[n - 1] = 0.0;
    for (int k = n - 2; k >= 0; k--) {
        y2[k] = y2[k] * y2[k + 1] + u[k];
    }
}

// Given the spline arrays (x, y, y2) compute interpolated value at xi.
double spline_eval(const vector<double>& x, const vector<double>& y, const vector<double>& y2, double xi) {
    int n = x.size();
    // Binary search for the right interval
    int klo = 0, khi = n - 1;
    while (khi - klo > 1) {
        int k = (khi + klo) >> 1;
        if (x[k] > xi)
            khi = k;
        else
            klo = k;
    }
    double h = x[khi] - x[klo];
    if (h == 0.0) {
        cerr << "Bad input to spline_eval: zero interval\n";
        exit(1);
    }
    double a = (x[khi] - xi) / h;
    double b = (xi - x[klo]) / h;
    double yi_val = a * y[klo] + b * y[khi] + ((a * a * a - a) * y2[klo] + (b * b * b - b) * y2[khi]) * (h * h) / 6.0;
    return yi_val;
}

// --- Main ---
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // --- Step 1: Read and extrapolate the kernel ---
    vector<double> kernel_time;   // original time values (every 2 dt)
    vector<double> kernel_value;  // corresponding kernel values

    vector<double> K_final(TOTAL_TIME, 0.0);  // final kernel array (dt=1)

    if (rank == 0) {
        ifstream kin("./carof/K.out");
        if (!kin) {
            cerr << "Error opening kernel file \"K.out\"\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        double t, val, junk;
        while (kin >> t >> val) {
            kernel_time.push_back(t);
            kernel_value.push_back(val);
        }
        kin.close();
        cout << "Rank 0: Kernel file read with " << kernel_time.size() << " points.\n";

        // Build cubic spline for interpolation using the original data.
        vector<double> spline_y2(kernel_time.size(), 0.0);
        compute_spline(kernel_time, kernel_value, spline_y2);

        // For t from 0 to 599, use spline interpolation.
        for (int t_i = 0; t_i < 600 && t_i < TOTAL_TIME; t_i++) {
            // Assume t_i lies within the original kernel_time range (kernel_time.back() should be >= 3998)
            K_final[t_i] = spline_eval(kernel_time, kernel_value, spline_y2, t_i);
        }

        // --- Exponential fit over t = 600 to 2500 ---
        int t_start = 600, t_end = 2500;
        double sumx = 0, sumy = 0, sumxx = 0, sumxy = 0;
        int count = 0;
        for (int t_i = t_start; t_i <= t_end; t_i++) {
            double Kt = spline_eval(kernel_time, kernel_value, spline_y2, t_i);
            if (Kt <= 0) continue; // only use positive values
            double logK = log(Kt);
            sumx += t_i;
            sumy += logK;
            sumxx += t_i * t_i;
            sumxy += t_i * logK;
            count++;
        }
        if (count < 2) {
            cerr << "Not enough points for exponential fit.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Linear regression: fit log(K) = intercept + slope * t.
        double slope = (count * sumxy - sumx * sumy) / (count * sumxx - sumx * sumx);
        double intercept = (sumy - slope * sumx) / count;
        // We want to model K(t) = A * exp(-b*t). Hence, set A = exp(intercept) and b = -slope.
        double A = exp(intercept);
        double b = -slope;
        cout << "Rank 0: Exponential fit parameters: A = " << A << ", decay constant (tau) = " << (1.0 / b)
             << "  (b = " << b << ")\n";

        // For t from 600 to TOTAL_TIME-1, use the exponential fit.
        for (int t_i = 600; t_i < TOTAL_TIME; t_i++) {
            K_final[t_i] = A * exp(-b * t_i);
        }
        // Write final kernel to file for checking.
        ofstream kout("K_extrapolated.txt");
        for (int t_i = 0; t_i < TOTAL_TIME; t_i++) {
            kout << t_i << " " << K_final[t_i] << "\n";
        }
        kout.close();
        cout << "Rank 0: Kernel extrapolation complete; written to \"K_extrapolated.txt\".\n";
    }

    // Broadcast final kernel array to all processes.
    MPI_Bcast(K_final.data(), TOTAL_TIME, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- Step 2: Read the trajectory file ---
    int num_particles = 0, num_frames = 0;
    vector<Frame> frames;
    if (rank == 0) {
        ifstream traj("cg.lammpstrj");
        if (!traj) {
            cerr << "Error opening trajectory file \"lammpstrj\"\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        string line;
        while (getline(traj, line)) {
            if (line.find("ITEM: TIMESTEP") != string::npos) {
                Frame frame;
                getline(traj, line); // timestep value
                frame.timestep = stoi(line);
                getline(traj, line); // ITEM: NUMBER OF ATOMS
                getline(traj, line); // number of atoms value
                num_particles = stoi(line);
                // Skip box bounds (assume 3 lines) and the atoms header (total 4 lines)
                for (int i = 0; i < 4; i++) getline(traj, line);
                getline(traj, line); // atoms header line
                // Read particle data (assumed to be num_particles lines)
                for (int i = 0; i < num_particles; i++) {
                    Particle p;
                    getline(traj, line);
                    istringstream iss(line);
                    // Expecting 11 columns: id, type, x, y, z, vx, vy, vz, fx, fy, fz
                    iss >> p.id >> p.type >> p.x >> p.y >> p.z >> p.vx >> p.vy >> p.vz >> p.fx >> p.fy >> p.fz;
                    frame.particles.push_back(p);
                }
                frames.push_back(frame);
                num_frames++;
                if (num_frames % 100 == 0)
                    cout << "Rank 0: Read " << num_frames << " frames.\n";
            }
        }
        traj.close();
        cout << "Rank 0: Trajectory reading complete. Total frames: " << num_frames
             << ", particles per frame: " << num_particles << "\n";
    }
    // Broadcast num_frames and num_particles
    MPI_Bcast(&num_frames, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&num_particles, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Rearrange the trajectory: For each particle, store its time series of momentum (m*v) and force.
    // We flatten the data into 1D arrays (size = num_particles * num_frames) for each component.
    vector<double> flat_mom_x, flat_mom_y, flat_mom_z;
    vector<double> flat_fx, flat_fy, flat_fz;
    if (rank == 0) {
        flat_mom_x.resize(num_particles * num_frames);
        flat_mom_y.resize(num_particles * num_frames);
        flat_mom_z.resize(num_particles * num_frames);
        flat_fx.resize(num_particles * num_frames);
        flat_fy.resize(num_particles * num_frames);
        flat_fz.resize(num_particles * num_frames);
        for (int t = 0; t < num_frames; t++) {
            for (int i = 0; i < num_particles; i++) {
                // Note: we assume particle id runs from 1 to num_particles.
                Particle &p = frames[t].particles[i];
                int idx = (p.id - 1) * num_frames + t;
                flat_mom_x[idx] = mass * p.vx;
                flat_mom_y[idx] = mass * p.vy;
                flat_mom_z[idx] = mass * p.vz;
                flat_fx[idx] = p.fx;
                flat_fy[idx] = p.fy;
                flat_fz[idx] = p.fz;
            }
        }
        // Free memory from frames
        frames.clear();
    } else {
        flat_mom_x.resize(num_particles * num_frames);
        flat_mom_y.resize(num_particles * num_frames);
        flat_mom_z.resize(num_particles * num_frames);
        flat_fx.resize(num_particles * num_frames);
        flat_fy.resize(num_particles * num_frames);
        flat_fz.resize(num_particles * num_frames);
    }
    // Broadcast flattened arrays to all processes.
    MPI_Bcast(flat_mom_x.data(), num_particles * num_frames, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_mom_y.data(), num_particles * num_frames, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_mom_z.data(), num_particles * num_frames, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_fx.data(), num_particles * num_frames, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_fy.data(), num_particles * num_frames, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_fz.data(), num_particles * num_frames, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- Step 3: Compute projected force via convolution ---
    // We need to compute for each particle and each component:
    //   conv(t) = sum_{s=0}^{t} K(t-s)*[m*v(s)]
    // We perform FFT convolution. For a particle time series of length L1 (num_frames) and a kernel of length L2 (TOTAL_TIME),
    // the full convolution length is L_conv = L1 + L2 - 1.
    int L1 = num_frames;
    int L2 = TOTAL_TIME;
    int conv_length = L1 + L2 - 1;
    int N_fft = 1;
    while (N_fft < conv_length) N_fft *= 2;
    if (rank == 0)
        cout << "Rank 0: FFT length = " << N_fft << "\n";

    // Precompute FFT of the kernel (padded to N_fft)
    vector<double> kernel_pad(N_fft, 0.0);
    for (int i = 0; i < L2; i++) {
        kernel_pad[i] = K_final[i];
    }
    fftw_complex* kernel_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N_fft / 2 + 1));
    fftw_plan plan_kernel = fftw_plan_dft_r2c_1d(N_fft, kernel_pad.data(), kernel_fft, FFTW_ESTIMATE);
    fftw_execute(plan_kernel);
    fftw_destroy_plan(plan_kernel);
    // Now kernel_fft holds the FFT of the kernel.

    // We will compute convolution for each momentum component and each particle.
    // Allocate output arrays for the convolution (only first L1 values are needed)
    vector<double> proj_conv_x(num_particles * L1, 0.0);
    vector<double> proj_conv_y(num_particles * L1, 0.0);
    vector<double> proj_conv_z(num_particles * L1, 0.0);

    // Distribute particles among MPI ranks
    int particles_per_rank = num_particles / size;
    int remainder = num_particles % size;
    int start_particle, end_particle;
    if (rank < remainder) {
        start_particle = rank * (particles_per_rank + 1);
        end_particle = start_particle + particles_per_rank;
    } else {
        start_particle = rank * particles_per_rank + remainder;
        end_particle = start_particle + particles_per_rank - 1;
    }
    cout << "Rank " << rank << " processing particles " << start_particle << " to " << end_particle << "\n";

    // Temporary buffers for FFT convolution
    // (We perform the convolution for each particle and for each component separately.)
    #pragma omp parallel for schedule(dynamic)
    for (int i = start_particle; i <= end_particle; i++) {
        // Allocate temporary arrays (re-used for each component)
        vector<double> P_pad(N_fft, 0.0);
        vector<double> conv_result(N_fft, 0.0);
        fftw_complex* P_fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * (N_fft / 2 + 1));
        fftw_plan plan_forward = fftw_plan_dft_r2c_1d(N_fft, P_pad.data(), P_fft, FFTW_ESTIMATE);
        fftw_plan plan_backward = fftw_plan_dft_c2r_1d(N_fft, P_fft, conv_result.data(), FFTW_ESTIMATE);

        // --- Process x component ---
        // Copy the momentum time series for particle i
        for (int t = 0; t < L1; t++) {
            P_pad[t] = flat_mom_x[i * L1 + t];
        }
        // Zero padding is already in P_pad.
        fftw_execute(plan_forward);
        // Multiply (pointwise) in Fourier space with kernel_fft
        for (int k = 0; k < N_fft / 2 + 1; k++) {
            double r = P_fft[k][0] * kernel_fft[k][0] - P_fft[k][1] * kernel_fft[k][1];
            double im = P_fft[k][0] * kernel_fft[k][1] + P_fft[k][1] * kernel_fft[k][0];
            P_fft[k][0] = r;
            P_fft[k][1] = im;
        }
        fftw_execute(plan_backward);
        // Normalize inverse FFT (FFTW does not normalize)
        for (int t = 0; t < L1; t++) {
            proj_conv_x[i * L1 + t] = conv_result[t] / N_fft;
        }

        // --- Process y component ---
        fill(P_pad.begin(), P_pad.end(), 0.0);
        fill(conv_result.begin(), conv_result.end(), 0.0);
        for (int t = 0; t < L1; t++) {
            P_pad[t] = flat_mom_y[i * L1 + t];
        }
        fftw_execute(plan_forward);
        for (int k = 0; k < N_fft / 2 + 1; k++) {
            double r = P_fft[k][0] * kernel_fft[k][0] - P_fft[k][1] * kernel_fft[k][1];
            double im = P_fft[k][0] * kernel_fft[k][1] + P_fft[k][1] * kernel_fft[k][0];
            P_fft[k][0] = r;
            P_fft[k][1] = im;
        }
        fftw_execute(plan_backward);
        for (int t = 0; t < L1; t++) {
            proj_conv_y[i * L1 + t] = conv_result[t] / N_fft;
        }

        // --- Process z component ---
        fill(P_pad.begin(), P_pad.end(), 0.0);
        fill(conv_result.begin(), conv_result.end(), 0.0);
        for (int t = 0; t < L1; t++) {
            P_pad[t] = flat_mom_z[i * L1 + t];
        }
        fftw_execute(plan_forward);
        for (int k = 0; k < N_fft / 2 + 1; k++) {
            double r = P_fft[k][0] * kernel_fft[k][0] - P_fft[k][1] * kernel_fft[k][1];
            double im = P_fft[k][0] * kernel_fft[k][1] + P_fft[k][1] * kernel_fft[k][0];
            P_fft[k][0] = r;
            P_fft[k][1] = im;
        }
        fftw_execute(plan_backward);
        for (int t = 0; t < L1; t++) {
            proj_conv_z[i * L1 + t] = conv_result[t] / N_fft;
        }

        fftw_destroy_plan(plan_forward);
        fftw_destroy_plan(plan_backward);
        fftw_free(P_fft);

        // Print progress every 100 particles processed (using OpenMP critical)
        if ((i - start_particle) % 100 == 0) {
            #pragma omp critical
            {
                cout << "Rank " << rank << " processed particle " << i << "\n";
            }
        }
    } // end parallel loop

    // Combine the original force with the convolution result:
    // Projected force = F + factor * (convolution result)
    vector<double> flat_proj_fx(num_particles * L1, 0.0);
    vector<double> flat_proj_fy(num_particles * L1, 0.0);
    vector<double> flat_proj_fz(num_particles * L1, 0.0);
    for (int i = start_particle; i <= end_particle; i++) {
        for (int t = 0; t < L1; t++) {
            int idx = i * L1 + t;
            flat_proj_fx[idx] = flat_fx[idx] + factor * proj_conv_x[idx];
            flat_proj_fy[idx] = flat_fy[idx] + factor * proj_conv_y[idx];
            flat_proj_fz[idx] = flat_fz[idx] + factor * proj_conv_z[idx];
        }
    }

    // Gather projected forces from all ranks onto rank 0.
    vector<double> all_proj_fx, all_proj_fy, all_proj_fz;
    if (rank == 0) {
        all_proj_fx.resize(num_particles * L1, 0.0);
        all_proj_fy.resize(num_particles * L1, 0.0);
        all_proj_fz.resize(num_particles * L1, 0.0);
    }
    MPI_Reduce(flat_proj_fx.data(), all_proj_fx.data(), num_particles * L1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(flat_proj_fy.data(), all_proj_fy.data(), num_particles * L1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(flat_proj_fz.data(), all_proj_fz.data(), num_particles * L1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    fftw_free(kernel_fft); // cleanup kernel FFT

    // --- Step 4: Write out the projected trajectory file ---
    if (rank == 0) {
        // We re-read the trajectory file to get positions and velocities.
        ifstream traj("cg.lammpstrj");
        if (!traj) {
            cerr << "Error opening trajectory file for output.\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ofstream pout("projected.lammpstrj");
        string line;
        int current_frame = 0;
        while(getline(traj, line)) {
            if (line.find("ITEM: TIMESTEP") != string::npos) {
                // Write header lines unchanged
                pout << line << "\n";
                getline(traj, line);
                pout << line << "\n"; // timestep value
                getline(traj, line);
                pout << line << "\n"; // number of atoms header
                getline(traj, line);
                pout << line << "\n"; // number of atoms value
                // Copy box bounds and atoms header (assume next 5 lines)
                for (int i = 0; i < 5; i++) {
                    getline(traj, line);
                    pout << line << "\n";
                }
                // Now process particle lines.
                for (int i = 0; i < num_particles; i++) {
                    getline(traj, line);
                    istringstream iss(line);
                    Particle p;
                    // Read first 8 columns: id, type, x, y, z, vx, vy, vz
                    iss >> p.id >> p.type >> p.x >> p.y >> p.z >> p.vx >> p.vy >> p.vz;
                    // Get projected force for this particle at this frame.
                    int idx = (p.id - 1) * L1 + current_frame;
                    double pf_x = all_proj_fx[idx];
                    double pf_y = all_proj_fy[idx];
                    double pf_z = all_proj_fz[idx];
                    // Write out the line with appended projected force columns.
                    pout << p.id << " " << p.type << " " << p.x << " " << p.y << " " << p.z << " "
                         << p.vx << " " << p.vy << " " << p.vz << " "
                         << pf_x << " " << pf_y << " " << pf_z << "\n";
                }
                current_frame++;
                cout << "Rank 0: Written frame " << current_frame << "\n";
            }
        }
        traj.close();
        pout.close();
        cout << "Rank 0: Projected trajectory written to \"projected_lammpstrj.txt\".\n";
    }

    MPI_Finalize();
    return 0;
}

