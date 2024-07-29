import os
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import imageio

# Custom CSS for background and styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://www.example.com/your-background-image.jpg");
        background-size: cover;
    }
    .sidebar .sidebar-content {
        background: url("https://www.example.com/your-sidebar-image.jpg");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class ModeloPsyllid:
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
        condiciones_iniciales=None,
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

        self.default_initial_conditions = [400, 200, 0, 0, 399, 1]
        self.condiciones_iniciales = (
            condiciones_iniciales
            if condiciones_iniciales is not None
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
            self.condiciones_iniciales,
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
            self.solution.t, self.solution.y[0], label="P_sanos(t)", linewidth=2.5
        )
        axs[0].plot(
            self.solution.t, self.solution.y[1], label="P_infectados(t)", linewidth=2.5
        )
        axs[0].set_xlabel("Meses")
        axs[0].set_ylabel("Población (P)")
        axs[0].set_title("Dinámica poblacional de Psyllid (P)")
        axs[0].legend()
        if ylimp is not None:
            axs[0].set_ylim(0, ylimp)

        axs[1].plot(
            self.solution.t, self.solution.y[4], label="T_sanos(t)", linewidth=2.5
        )
        axs[1].plot(
            self.solution.t, self.solution.y[5], label="T_infectados(t)", linewidth=2.5
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
        arboles_infectados = self.solution.y[5]  # Ti está en el índice 5
        peak_index = np.argmax(arboles_infectados)
        peak_time = self.solution.t[peak_index]
        peak_value = arboles_infectados[peak_index]
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
                self.params[1] = c  # Actualizar c en la lista de parámetros
                self.params[4] = mua  # Actualizar mua en la lista de parámetros
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
                self.params[2] = lambdu  # Actualizar lambdu en la lista de parámetros
                self.params[4] = mua  # Actualizar mua en la lista de parámetros
                self.solve()
                peak_time, _ = self.get_peak_infected_time()
                peak_times[i, j] = peak_time

        return lambdu_values, mua_values, peak_times


def crear_gif(
    modelo,
    nombre_parametro,
    inicio_parametro,
    fin_parametro,
    paso_parametro,
    condiciones_iniciales,
):
    if nombre_parametro not in modelo.param_names:
        raise ValueError(f"Nombre de parámetro inválido: {nombre_parametro}")
    indice_parametro = modelo.param_names.index(nombre_parametro)
    valores_parametro = np.arange(
        inicio_parametro, fin_parametro + paso_parametro, paso_parametro
    )
    nombres_archivos = []
    for valor_parametro in valores_parametro:
        # Crear una nueva instancia del modelo con parámetros actualizados y condiciones iniciales
        nuevos_parametros = modelo.params.copy()
        nuevos_parametros[indice_parametro] = valor_parametro
        nuevo_modelo = ModeloPsyllid(
            *nuevos_parametros, condiciones_iniciales=condiciones_iniciales
        )
        nuevo_modelo.solve()

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].plot(
            nuevo_modelo.solution.t,
            nuevo_modelo.solution.y[0],
            label="P_sanos(t)",
            linewidth=2.5,
        )
        axs[0].plot(
            nuevo_modelo.solution.t,
            nuevo_modelo.solution.y[1],
            label="P_infectados(t)",
            linewidth=2.5,
        )
        axs[0].set_xlabel("Meses")
        axs[0].set_ylabel("Población (P)")
        axs[0].set_title(
            f"Dinámica poblacional de Psyllid (P) - {nombre_parametro}={valor_parametro:.2f}"
        )
        axs[0].legend()
        axs[1].plot(
            nuevo_modelo.solution.t,
            nuevo_modelo.solution.y[4],
            label="T_sanos(t)",
            linewidth=2.5,
        )
        axs[1].plot(
            nuevo_modelo.solution.t,
            nuevo_modelo.solution.y[5],
            label="T_infectados(t)",
            linewidth=2.5,
        )
        axs[1].set_xlabel("Meses")
        axs[1].set_ylabel("Población (T)")
        axs[1].set_title(
            f"Dinámica poblacional de Árboles (T) - {nombre_parametro}={valor_parametro:.2f}"
        )
        axs[1].legend()
        plt.tight_layout()
        nombre_archivo = f"frame_{nombre_parametro}_{valor_parametro:.2f}.png"
        nombres_archivos.append(nombre_archivo)
        plt.savefig(nombre_archivo)
        plt.close()
        print(f"Archivo guardado: {nombre_archivo}")

    # Revertir la lista de nombres de archivos para crear el bucle inverso
    nombres_archivos += nombres_archivos[::-1]

    # Crear un nombre de archivo único para el GIF
    nombre_gif = f"dinamicas_poblacionales_{nombre_parametro}.gif"

    with imageio.get_writer(nombre_gif, mode="I", duration=0.5) as writer:
        for nombre_archivo in nombres_archivos:
            try:
                imagen = imageio.imread(nombre_archivo)
                writer.append_data(imagen)
            except FileNotFoundError:
                print(f"Archivo no encontrado: {nombre_archivo}")

    # Limpiar archivos temporales
    for nombre_archivo in nombres_archivos:
        try:
            os.remove(nombre_archivo)
            print(f"Archivo eliminado: {nombre_archivo}")
        except FileNotFoundError:
            print(f"Archivo no encontrado para eliminar: {nombre_archivo}")

    # Mostrar el GIF en Streamlit
    st.image(nombre_gif)

    # Opcionalmente, eliminar el archivo GIF después de mostrarlo
    os.remove(nombre_gif)
    print(f"Archivo GIF eliminado: {nombre_gif}")


def main():
    st.title("Modelo de Dinámica Poblacional de Psyllid")

    st.sidebar.header("Parámetros del Modelo")
    with st.sidebar.form(key="params_form"):
        # Model parameter sliders
        c = st.slider("c: Preferencia por infectados", 0.1, 0.9, 0.5, 0.01)
        mua = st.slider(
            "μa: Tasa de muerte de psílidos con plantas trampa", 0.1, 1.0, 0.75, 0.01
        )
        lambdu = st.slider("λ: Tasa de infección de psílidos", 0.1, 0.9, 0.75, 0.01)
        gamma = st.slider("γ: Tasa de infección de árboles", 0.01, 0.9, 0.1, 0.01)

        # Initial conditions input
        st.header("Condiciones Iniciales")
        Pu_init = st.number_input("Psílidos sanos (Pu)", value=400)
        Pi_init = st.number_input("Psílidos infectados (Pi)", value=200)
        Iu_init = st.number_input("Larvas sanas (Iu)", value=0)
        Ii_init = st.number_input("Larvas infectadas (Ii)", value=0)
        Tu_init = st.number_input("Árboles sanos (Tu)", value=399)
        Ti_init = st.number_input("Árboles infectados (Ti)", value=1)

        submitted = st.form_submit_button("Ejecutar Modelo")

    initial_conditions = [Pu_init, Pi_init, Iu_init, Ii_init, Tu_init, Ti_init]

    modelo = ModeloPsyllid(
        c=c,
        mua=mua,
        lambdu=lambdu,
        gamma=gamma,
        condiciones_iniciales=initial_conditions,
    )

    if submitted:
        with st.spinner("Resolviendo el modelo..."):
            modelo.solve()
            modelo.plot()
            peak_time, peak_value = modelo.get_peak_infected_time()
            st.success(
                f"El número máximo de árboles infectados se alcanza en el mes {peak_time:.2f} con un valor de {peak_value:.2f}."
            )

    # GIF creation section
    st.sidebar.header("Crear GIF")
    with st.sidebar.form(key="gif_form"):
        nombre_parametro = st.selectbox(
            "Elige el parámetro a variar", modelo.param_names
        )
        inicio_parametro = st.number_input(
            f"Valor inicial para {nombre_parametro}", value=0.1, step=0.01
        )
        fin_parametro = st.number_input(
            f"Valor final para {nombre_parametro}", value=0.99, step=0.01
        )
        paso_parametro = st.number_input(
            f"Tamaño del paso para {nombre_parametro}", value=0.025, step=0.005
        )

        generate_gif = st.form_submit_button("Generar GIF")

    if generate_gif:
        with st.spinner("Generando GIF... Esto puede tardar un momento."):
            crear_gif(
                modelo,
                nombre_parametro,
                inicio_parametro,
                fin_parametro,
                paso_parametro,
                initial_conditions,
            )
        st.success("¡GIF generado con éxito!")


if __name__ == "__main__":
    main()
