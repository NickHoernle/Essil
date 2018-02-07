functions {
  real next_water(vector curr_state, vector thetas, real tree_drink_rate, int biome) {

    real next_water_;

    real evaporation_rate = thetas[1];
    int desert = 2;
    int plains = 3;
    int jungle = 4;
    int wetlands = 5;
    int reservoir = 6;

    next_water_ = 0;

    if (biome == 1) {
      // other water
      next_water_ = (curr_state[biome] +
                    tree_drink_rate*(curr_state[desert+5] + curr_state[plains+5] + curr_state[jungle+5] + curr_state[wetlands+5]) + //water from tree evaporation
                    evaporation_rate*(curr_state[desert] + curr_state[plains] + curr_state[jungle] + curr_state[wetlands]) + // water from normal evaporation
                    (thetas[desert+5]*curr_state[desert] + thetas[plains+5]*curr_state[plains] + thetas[jungle+5]*curr_state[jungle] + thetas[wetlands+5]*curr_state[wetlands]) + // water released from biomes from users
                    thetas[reservoir]*curr_state[reservoir] - // water released from reservoir from users
                    (thetas[desert]*curr_state[1] + thetas[plains]*curr_state[1] + thetas[jungle]*curr_state[1] + thetas[wetlands]*curr_state[1])); // water directed into biomes
    }
    else if (biome == 6) {
      // reservoir water
      next_water_ = (curr_state[biome] - thetas[biome]*curr_state[biome]);// + // water_out user
                     // 0.75*tree_drink_rate*(curr_state[wetlands+5]) +
                     // 0.25*tree_drink_rate*(curr_state[jungle+5]));
    }
    else {
      // biome water
      next_water_ = (curr_state[biome] +                    // current_water
                    thetas[biome]*curr_state[1] -           // water_in (water from waterfall) user
                    evaporation_rate*curr_state[biome] -    // water_out evaporation
                    tree_drink_rate*curr_state[biome+5] -   // water_out trees
                    thetas[biome+5]*curr_state[biome]);     // water_out (water from the current state) user
    }

    return next_water_;
  }

  vector get_error(vector current_state, vector previous_state, vector emission_param, real evaporation, real tree_drink, int N) {

    vector[N] err;
    for (n in 2:N)
      // the error is the actual water - the predicted water using the previous state and the current number of trees
      err[n] = current_state[n] - next_water(previous_state, append_row(evaporation, emission_param), tree_drink, n);
    err[1] = next_water(previous_state, append_row(evaporation, emission_param), tree_drink, 1) - current_state[1];
    return err;
  }

  vector get_expected_rain(vector clouds, real rain_rate, int N) {

    vector[N] rain;

    for (n in 2:N)
      rain[n] = rain_rate * clouds[n];
    rain[1] = rain_rate * max(clouds); // accounted for with the evaporation above

    return rain;
  }
}

data {
  int<lower=1> N;                // number of dimensions
  int<lower=1> M;                // number of chains

  int<lower=0> T;                // number of data points
  vector[16] y[T];               // output

  int<lower=-1,upper=M> z[T];    // supervised assignments
  int<lower=0> unsup_count;
  real<lower=0, upper=1> theta_trans;

  real<lower=0> lambda;
  real<lower=0> precision;
}

transformed data {
  simplex[M] phi_trans[M];

  for (m in 1:M-1){
    phi_trans[m] = rep_vector(0,M);
    phi_trans[m,m + 1] = theta_trans;
    phi_trans[m,m] = 1-theta_trans;
  }
  phi_trans[M] = rep_vector(0,M);
  phi_trans[M,M] = 1;
}

parameters {
  // real<lower=0> r;
  real<lower=0, upper=1> water_flow_rate;
  // real<lower=0, upper=1> waterfall_rate;
  real<lower=0, upper=1> evaporation;
  real<lower=0, upper=1> tree_drink;

  simplex[4] water_in[M];
  vector<lower=-20, upper=20>[5] water_out[M];
  simplex[4] plan_prior;

  real<lower=1e-10> r; // noise distribution for each biome
}

transformed parameters{
  vector<lower=0, upper=1>[9] theta_emis[M];

  for (m in 1:M){
    for (i in 1:4)
      theta_emis[m][i] = water_flow_rate * water_in[m][i];
    for (i in 1:5)
      theta_emis[m][4+i] = water_flow_rate * (1/(1+exp(-(water_out[m][i] - 10))));
  }
}

model {
  evaporation ~ beta(1,15);
  tree_drink ~ beta(1,15);
  // waterfall_rate ~ beta(1,15);
  water_flow_rate ~ beta(1,15);

  r ~ cauchy(0,1/precision);

  plan_prior ~ dirichlet(to_vector([1,1,1,1]));

  for (m in 1:M) {

    water_in[m] ~ dirichlet(16*plan_prior);
    for (i in 1:5)
      // penalise the model for using these parameters
      water_out[m][i] ~ double_exponential(0,1/lambda);
  }

  {
    vector[N] err;
    vector[N] expected_rain;
    vector[2] lp;

    int last_state;
    int last_time_step;

    real gamma[T, M];

    last_state = 1;

    gamma[1,1] = 0;
    for (m in 2:M)
      gamma[1,m] = negative_infinity();

    for (t in 2:T){

      expected_rain = get_expected_rain(y[t][11:16], r, N);

      if (z[t] == -1) {
      // this is the unsupervised part of the model
      // given the previous state. We could be in the same state as the previous timestep or we could have
      // moved to the next state according to the transition parameters.
        for (m in 0:1) {
          for (j in 0:1) {
            err = get_error(y[t], y[t-1], theta_emis[last_state+m], evaporation, tree_drink, N);
            lp[j+1] = gamma[t-1, last_state+j] + log(phi_trans[last_state+j, last_state+m]) + normal_lpdf(err | rep_vector(0, N), expected_rain);
          }
          gamma[t, last_state+m] = log_sum_exp(lp);
          if (t >= last_time_step + unsup_count){ target += log_sum_exp(lp); }
        }

      } else {
      // here we are supervised and thus can sample the transition matrices for the hidden chain
        err = get_error(y[t], y[t-1], theta_emis[z[t]], evaporation, tree_drink, N);
        // there is the probability of rain in the biomes
        err ~ normal(rep_vector(0, N), expected_rain);
        last_state = z[t];
        last_time_step = t;

        for (m in 1:M)
          gamma[t,m] = negative_infinity();
        gamma[t,z[t]] = 0;
      }
    }
  }
}

generated quantities {

  int<lower=1,upper=M> s[T];
  {
    int back_ptr[T, M];
    vector[N] err;
    vector[N-1] expected_rain;
    real best_logp[T, M];
    real best_total_logp;

    for (m in 2:M)
      best_logp[1, m] = negative_infinity();
    best_logp[1, 1] = 0;

    for (t in 2:T) {

      expected_rain = get_expected_rain(y[t][11:16], r, N)[2:];

      for (m in 1:M) {
        best_logp[t, m] = negative_infinity();

        for (j in 1:M) {
          real logp;

          if (phi_trans[j, m] == 0) { logp = negative_infinity(); }
          else {

            err = get_error(y[t], y[t-1], theta_emis[m], evaporation, tree_drink, N);
            logp = best_logp[t-1, j] + log(phi_trans[j, m]) + normal_lpdf(err[2:] | rep_vector(0, N-1), expected_rain);

          }
          if (logp > best_logp[t, m]) {
            back_ptr[t, m] = j;
            best_logp[t, m] = logp;
          }
        }
      }
    }

    s[T] = M; // We know that the final chain is active in the final portion
    for (t in 1:(T - 1))
      s[T - t] = back_ptr[T - t + 1, s[T - t + 1]];
  }
}
