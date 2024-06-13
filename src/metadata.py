class MetaData:
    def __init__(self):
        self.out_dir = ""
        self.lambdas = []

        self.nr_x = 0
        self.nr_y = 0

        self.lattice_constant = 0.0
        self.fill_factor = 0.0

        self.tilt_angle = 0.0

        self.phi_out_start = 0.0
        self.phi_out_end = 0.0
        self.theta_out_start = 0.0
        self.theta_out_end = 0.0
        self.out_step_size = 0.0

        self.phi_in_start = 0.0
        self.phi_in_end = 0.0
        self.theta_in_start = 0.0
        self.theta_in_end = 0.0
        self.in_step_size = 0.0

    def __str__(self):
        lambdas_string = " ".join(map(str, self.lambdas))
        ret_string = "out_dir: " + self.out_dir + \
                     "\nlambdas: " + lambdas_string + \
                     "\nnr_x: " + str(self.nr_x) + \
                     "\nnr_y: " + str(self.nr_y) + \
                     "\nlattice_constant: " + str(self.lattice_constant) + \
                     "\nfill_factor: " + str(self.fill_factor) + \
                     "\ntilt_angle: " + str(self.tilt_angle) + \
                     "\nphi_out_start: " + str(self.phi_out_start) + \
                     "\nphi_out_end: " + str(self.phi_out_end) + \
                     "\ntheta_out_start: " + str(self.theta_out_start) + \
                     "\ntheta_out_end: " + str(self.theta_out_end) + \
                     "\nout_step_size: " + str(self.out_step_size) + \
                     "\nphi_in_start: " + str(self.phi_in_start) + \
                     "\nphi_in_end: " + str(self.phi_in_end) + \
                     "\ntheta_in_start: " + str(self.theta_in_start) + \
                     "\ntheta_in_end: " + str(self.theta_in_end) + \
                     "\nin_step_size: " + str(self.in_step_size)
        return ret_string

