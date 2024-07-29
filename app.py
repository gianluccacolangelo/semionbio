import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import imageio


class PsyllidModel:
    def __init__(
        self,
        a=2.34,
        c=0.75,
        lambdu=0.75,
        p=1.22,
        mua=0.75,
        f=100,
        v=0.75,
        omega=np.pi / 8,
        w=45,
        q=0.5,
        mui=0.105,
        g=3,
        gamma=0.1,
        beta=0.04,
        se_curan=0,
        initial_conditions=None,
        t_span=[0, 350],
    ):

        self.params = [
            a,
            c,
            lambdu,
            p,
            mua,
            f,
            v,
            omega,
            w,
            q,
            mui,
            g,
            gamma,
            beta,
            se_curan,
        ]

        self.param_names = [
            "a",
            "c",
            "lambdu",
            "p",
            "mua",
            "f",
            "v",
            "omega",
            "w",
            "q",
            "mui",
            "g",
            "gamma",
            "beta",
            "se_curan",
        ]

        self.default_initial_conditions = [1e7, 1e2, 0, 0, 13000, 0]
        self.initial_conditions = (
            initial_conditions
            if initial_conditions is not None
            else self.default_initial_conditions
        )

        self.t_span = t_span
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], 1000)
        self.solution = None

    def model(self, t, y):
        Pu, Pi, Iu, Ii, Tu, Ti = y
        a, c, lambdu, p, mua, f, v, omega, w, q, mui, g, gamma, beta, se_curan = (
            self.params
        )

        dPu_dt = a * Iu - (lambdu * (1 - c) * Ti * Pu) / (Ti + Tu) - mua * Pu
        dPi_dt = a * Ii + (lambdu * c * Ti * Pu) / (Ti + Tu) - mua * Pi
        dIu_dt = (
            w
            * (Pu + Pi)
            * (1 - c)
            * (Tu / (Ti + Tu))
            * (1 - (Iu + Ii) / (f * (1 + v * np.sin(omega * t)) * (Ti + Tu)))
            + w
            * (Pu + Pi)
            * c
            * (1 - q)
            * (Ti / (Ti + Tu))
            * (1 - (Iu + Ii) / (f * (1 + v * np.sin(omega * t)) * (Ti + Tu)))
            - a * Iu
            - mui * Iu
        )
        dIi_dt = (
            w
            * (Pu + Pi)
            * c
            * (Ti / (Ti + Tu))
            * q
            * (1 - (Iu + Ii) / (f * (1 + v * np.sin(omega * t)) * (Ti + Tu)))
            - a * Ii
            - mui * Ii
        )
        dTu_dt = g - (gamma * Pi * Tu) / (Pi + Pu)
        dTi_dt = (gamma * Pi * Tu) / (Pi + Pu) - beta * Ti

        return [dPu_dt, dPi_dt, dIu_dt, dIi_dt, dTu_dt, dTi_dt]

    def solve(self):
        self.solution = solve_ivp(
            self.model,
            self.t_span,
            self.initial_conditions,
            t_eval=self.t_eval,
            method="LSODA",
        )
        return self.solution

    def plot(self, ylimp=None, ylimt=None):
        if self.solution is None:
            raise ValueError(
                "El modelo no ha sido resuelto aún. Llamá 'model.solve()' primero."
            )

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].plot(
            self.solution.t, self.solution.y[0], label="P_uninfected(t)", linewidth=2.5
        )
        axs[0].plot(
            self.solution.t, self.solution.y[1], label="P_infected(t)", linewidth=2.5
        )
        axs[0].set_xlabel("Meses")
        axs[0].set_ylabel("Población (P)")
        axs[0].set_title("Dinámica poblacional de Psyllid (P)")
        axs[0].legend()
        if ylimp is not None:
            axs[0].set_ylim(0, ylimp)

        axs[1].plot(
            self.solution.t, self.solution.y[4], label="T_uninfected(t)", linewidth=2.5
        )
        axs[1].plot(
            self.solution.t, self.solution.y[5], label="T_infected(t)", linewidth=2.5
        )
        axs[1].set_xlabel("Meses")
        axs[1].set_ylabel("Población (T)")
        axs[1].set_title("Dinámica poblacional de Árboles (T)")
        axs[1].legend()
        if ylimt is not None:
            axs[1].set_ylim(0, ylimt)

        plt.tight_layout()
        st.pyplot(fig)

    def get_peak_infected_time(self):
        if self.solution is None:
            raise ValueError(
                "El modelo no ha sido resuelto aún. Llamá 'model.solve()' primero."
            )
        infected_trees = self.solution.y[5]  # Ti is at index 5
        peak_index = np.argmax(infected_trees)
        peak_time = self.solution.t[peak_index]
        peak_value = infected_trees[peak_index]
        return peak_time, peak_value

    def run_c_mua_range(
        self,
        c_start=0.3,
        c_end=0.9,
        c_step=0.025,
        mua_start=0.2,
        mua_end=1.0,
        mua_step=0.025,
    ):
        c_values = np.arange(c_start, c_end + c_step, c_step)
        mua_values = np.arange(mua_start, mua_end + mua_step, mua_step)
        peak_times = np.zeros((len(c_values), len(mua_values)))

        for i, c in enumerate(c_values):
            for j, mua in enumerate(mua_values):
                self.params[1] = c  # Update c in the parameters list
                self.params[4] = mua  # Update mua in the parameters list
                self.solve()
                peak_time, _ = self.get_peak_infected_time()
                peak_times[i, j] = peak_time

        return c_values, mua_values, peak_times

    def run_lambdu_mua_range(
        self,
        lambdu_start=0.1,
        lambdu_end=0.9,
        lambdu_step=0.025,
        mua_start=0.2,
        mua_end=1.0,
        mua_step=0.025,
    ):
        lambdu_values = np.arange(lambdu_start, lambdu_end + lambdu_step, lambdu_step)
        mua_values = np.arange(mua_start, mua_end + mua_step, mua_step)
        peak_times = np.zeros((len(lambdu_values), len(mua_values)))

        for i, lambdu in enumerate(lambdu_values):
            for j, mua in enumerate(mua_values):
                self.params[2] = lambdu  # Update lambdu in the parameters list
                self.params[4] = mua  # Update mua in the parameters list
                self.solve()
                peak_time, _ = self.get_peak_infected_time()
                peak_times[i, j] = peak_time

        return lambdu_values, mua_values, peak_times


def create_gif(
    model, param_name, param_start, param_end, param_step, initial_conditions
):
    if param_name not in model.param_names:
        raise ValueError(f"Invalid parameter name: {param_name}")
    param_index = model.param_names.index(param_name)
    param_values = np.arange(param_start, param_end + param_step, param_step)
    filenames = []
    for param_value in param_values:
        # Create a new model instance with updated parameters and initial conditions
        new_params = model.params.copy()
        new_params[param_index] = param_value
        new_model = PsyllidModel(*new_params, initial_conditions=initial_conditions)
        new_model.solve()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].plot(
            new_model.solution.t,
            new_model.solution.y[0],
            label="P_uninfected(t)",
            linewidth=2.5,
        )
        axs[0].plot(
            new_model.solution.t,
            new_model.solution.y[1],
            label="P_infected(t)",
            linewidth=2.5,
        )
        axs[0].set_xlabel("Meses")
        axs[0].set_ylabel("Población (P)")
        axs[0].set_title(
            f"Dinámica poblacional de Psyllid (P) - {param_name}={param_value:.2f}"
        )
        axs[0].legend()
        axs[1].plot(
            new_model.solution.t,
            new_model.solution.y[4],
            label="T_uninfected(t)",
            linewidth=2.5,
        )
        axs[1].plot(
            new_model.solution.t,
            new_model.solution.y[5],
            label="T_infected(t)",
            linewidth=2.5,
        )
        axs[1].set_xlabel("Meses")
        axs[1].set_ylabel("Población (T)")
        axs[1].set_title(
            f"Dinámica poblacional de Árboles (T) - {param_name}={param_value:.2f}"
        )
        axs[1].legend()
        plt.tight_layout()
        filename = f"frame_{param_name}_{param_value:.2f}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()
        print(f"Saved frame: {filename}")

    # Reverse the filenames list to create the reverse loop
    filenames += filenames[::-1]

    # Create a unique filename for the GIF
    gif_filename = f"population_dynamics_{param_name}.gif"

    with imageio.get_writer(gif_filename, mode="I", duration=0.5) as writer:
        for filename in filenames:
            try:
                image = imageio.imread(filename)
                writer.append_data(image)
            except FileNotFoundError:
                print(f"File not found: {filename}")

    # Cleanup temporary files
    for filename in filenames:
        try:
            os.remove(filename)
            print(f"Removed file: {filename}")
        except FileNotFoundError:
            print(f"File not found for removal: {filename}")

    # Display the GIF in Streamlit
    st.image(gif_filename)

    # Optionally, remove the GIF file after displaying
    os.remove(gif_filename)
    print(f"Removed GIF file: {gif_filename}")


def main():
    st.title("Psyllid Population Dynamics Model")
    st.sidebar.header("Model Parameters")

    # Model parameter sliders
    c = st.sidebar.slider("c: Preferencia por infectados", 0.1, 0.9, 0.5, 0.01)
    mua = st.sidebar.slider(
        "μa: Tasa de muerte de psílidos con plantas trampa", 0.1, 1.0, 0.75, 0.01
    )
    lambdu = st.sidebar.slider("λ: Tasa de infección de psílidos", 0.1, 0.9, 0.75, 0.01)
    gamma = st.sidebar.slider("γ: Tasa de infección de árboles", 0.01, 0.9, 0.1, 0.01)

    # Default values for other parameters
    a = 2.34
    p = 1.22
    f = 100
    v = 0.75
    omega = np.pi / 8
    w = 45
    q = 0.5
    mui = 0.105
    g = 3
    beta = 0.04
    se_curan = 0.0

    # Initial conditions input
    st.sidebar.header("Initial Conditions")
    Pu_init = st.sidebar.number_input("Psílidos sanos (Pu)", value=1e7, format="%e")
    Pi_init = st.sidebar.number_input(
        "Psílidos infectados (Pi)", value=1e2, format="%e"
    )
    Iu_init = st.sidebar.number_input("Larvas sanas (Iu)", value=0.0, format="%e")
    Ii_init = st.sidebar.number_input("Larvas infectadas (Ii)", value=0.0, format="%e")
    Tu_init = st.sidebar.number_input("Árboles sanos (Tu)", value=13000.0)
    Ti_init = st.sidebar.number_input("Árboles infectados (Ti)", value=0.0)

    initial_conditions = [Pu_init, Pi_init, Iu_init, Ii_init, Tu_init, Ti_init]

    model = PsyllidModel(
        a=a,
        c=c,
        lambdu=lambdu,
        p=p,
        mua=mua,
        f=f,
        v=v,
        omega=omega,
        w=w,
        q=q,
        mui=mui,
        g=g,
        gamma=gamma,
        beta=beta,
        se_curan=se_curan,
        initial_conditions=initial_conditions,
    )

    if st.sidebar.button("Run Model", key="run_model_button"):
        model.solve()
        model.plot()
        peak_time, peak_value = model.get_peak_infected_time()
        st.write(
            f"The maximum number of infected trees is reached at month {peak_time:.2f} with a value of {peak_value:.2f}."
        )

    # GIF creation section
    st.sidebar.header("Create GIF")
    param_name = st.sidebar.selectbox("Choose parameter to vary", model.param_names)
    param_start = st.sidebar.number_input(
        f"Start value for {param_name}", value=0.1, step=0.01
    )
    param_end = st.sidebar.number_input(
        f"End value for {param_name}", value=0.99, step=0.01
    )
    param_step = st.sidebar.number_input(
        f"Step size for {param_name}", value=0.025, step=0.005
    )

    if st.sidebar.button("Generate GIF", key="generate_gif_button"):
        with st.spinner("Generating GIF... This may take a moment."):
            create_gif(
                model,
                param_name,
                param_start,
                param_end,
                param_step,
                initial_conditions,
            )
        st.success("GIF generated successfully!")


if __name__ == "__main__":
    main()
