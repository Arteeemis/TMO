import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Заголовок приложения
st.title("📚 Прогнозирование успеваемости студентов")
st.markdown(
    """
Это приложение использует модель Lasso регрессии для предсказания итоговых оценок студентов 
на основе их привычек и образа жизни. Модель достигла R2 = 0.9 на тестовых данных.
"""
)


# Загрузка и предобработка данных
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("student_habits_performance.csv")
        # Удаляем ID если есть
        if "student_id" in data.columns:
            data = data.drop("student_id", axis=1)

        # Добавляем вспомогательные признаки как в оригинальном коде
        data["total_screen_time"] = data["social_media_hours"] + data["netflix_hours"]
        data["study_leisure_ratio"] = data["study_hours_per_day"] / (
            data["total_screen_time"] + 1e-6
        )

        return data
    except FileNotFoundError:
        st.error("Файл данных не найден. Пожалуйста, проверьте путь к файлу.")
        return None


data = load_data()

if data is not None:
    # Разделение данных
    X = data.drop("exam_score", axis=1)
    y = data["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Определение числовых и категориальных признаков (как в оригинальном коде)
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Добавляем числовые признаки с малым числом уникальных значений в категориальные
    maybe_cat = [col for col in numerical_cols if len(X[col].unique()) <= 5]
    categorical_cols.extend(maybe_cat)
    numerical_cols = [col for col in numerical_cols if col not in maybe_cat]

    # Создаем конвейер для обработки данных
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Обучение модели с оптимальными параметрами (из GridSearchCV)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Lasso(alpha=0.001, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)

    # Предсказания и метрики
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Отображение метрик
    st.header("📊 Метрики модели")
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", f"{mae:.2f}")
    col2.metric("Mean Squared Error", f"{mse:.2f}")
    col3.metric("R2 Score", f"{r2:.2f}")

    # Визуализация предсказаний vs реальных значений
    st.header("📈 Сравнение предсказаний и реальных значений")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    ax.set_xlabel("Реальные оценки")
    ax.set_ylabel("Предсказанные оценки")
    ax.set_title("Предсказания vs Реальные значения")
    st.pyplot(fig)

    # Важность признаков
    st.header("🔍 Важность признаков")
    try:
        # Получаем имена признаков после преобразования
        feature_names_num = numerical_cols
        ohe = (
            model.named_steps["preprocessor"]
            .named_transformers_["cat"]
            .named_steps["onehot"]
        )
        feature_names_cat = ohe.get_feature_names_out(categorical_cols)
        feature_names = np.concatenate([feature_names_num, feature_names_cat])

        # Получаем коэффициенты Lasso
        importances = model.named_steps["regressor"].coef_

        # Создаем DataFrame для визуализации
        importance_df = (
            pd.DataFrame({"Признак": feature_names, "Важность": importances})
            .sort_values("Важность", key=np.abs, ascending=False)
            .head(15)
        )

        # Визуализация
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Важность", y="Признак", data=importance_df, ax=ax)
        ax.set_title("Топ-15 важнейших признаков")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Не удалось визуализировать важность признаков: {str(e)}")

    # Предсказание для новых данных
    st.header("🔮 Сделать предсказание")
    st.markdown("Введите параметры студента для предсказания его итоговой оценки:")

    # Создаем форму для ввода данных
    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Возраст", min_value=15, max_value=30, value=20)
            study_hours = st.number_input(
                "Часы учебы в день", min_value=0.0, max_value=24.0, value=4.0
            )
            social_media_hours = st.number_input(
                "Часы в соцсетях", min_value=0.0, max_value=24.0, value=2.0
            )
            netflix_hours = st.number_input(
                "Часы просмотра Netflix", min_value=0.0, max_value=24.0, value=1.0
            )
            sleep_hours = st.number_input(
                "Часы сна", min_value=0.0, max_value=24.0, value=8.0
            )
            attendance = st.slider("Посещаемость (%)", 0, 100, 85)
            diet_quality = st.slider("Качество питания (1-10)", 1, 10, 7)

        with col2:
            exercise_freq = st.slider("Частота занятий спортом (раз в неделю)", 0, 7, 3)
            mental_health = st.slider("Уровень ментального здоровья (1-10)", 1, 10, 7)
            physical_health = st.slider("Уровень физического здоровья (1-10)", 1, 10, 7)
            part_time_job = st.selectbox("Подработка", ["Нет", "Да"])
            parental_education = st.selectbox(
                "Образование родителей", ["Среднее", "Высшее", "Аспирантура"]
            )
            internet_quality = st.selectbox(
                "Качество интернета", ["Плохое", "Среднее", "Хорошее"]
            )
            extracurricular = st.selectbox(
                "Участие во внеурочных занятиях", ["Нет", "Да"]
            )
            gender = st.selectbox("Пол", ["Мужской", "Женский"])

        submitted = st.form_submit_button("Предсказать оценку")

        if submitted:
            # Создаем DataFrame с введенными данными
            input_data = {
                "age": age,
                "study_hours_per_day": study_hours,
                "social_media_hours": social_media_hours,
                "netflix_hours": netflix_hours,
                "sleep_hours": sleep_hours,
                "attendance_percentage": attendance,
                "diet_quality": diet_quality,
                "exercise_frequency": exercise_freq,
                "mental_health_rating": mental_health,
                "physical_health_rating": physical_health,
                "part_time_job": 1 if part_time_job == "Да" else 0,
                "parental_education_level": parental_education,
                "internet_quality": internet_quality,
                "extracurricular_participation": 1 if extracurricular == "Да" else 0,
                "gender": gender,
                # Автоматически вычисляем дополнительные признаки
                "total_screen_time": social_media_hours + netflix_hours,
                "study_leisure_ratio": study_hours
                / ((social_media_hours + netflix_hours) + 1e-6),
            }

            # Создаем DataFrame с правильным порядком столбцов
            input_df = pd.DataFrame([input_data])

            # Убедимся, что все столбцы в правильном порядке
            input_df = input_df[X.columns]

            # Делаем предсказание
            prediction = model.predict(input_df)[0]
            if prediction > 100:
                prediction = 100

            # Отображаем результат
            st.success(f"Предсказанная итоговая оценка: {prediction:.1f} из 100")

            # Дополнительная информация
            st.markdown("### Рекомендации для улучшения успеваемости:")
            if study_hours < 3:
                st.warning(
                    "⚠️ Увеличьте время учебы. Рекомендуется минимум 3 часа в день."
                )
            if (social_media_hours + netflix_hours) > 4:
                st.warning(
                    "⚠️ Слишком много времени на развлечения. Попробуйте сократить."
                )
            if sleep_hours < 7:
                st.warning(
                    "⚠️ Недостаточно сна. Для лучшей успеваемости нужно 7-9 часов сна."
                )
            if mental_health < 5:
                st.warning(
                    "⚠️ Низкий уровень ментального здоровья. Обратитесь за помощью."
                )
            if attendance < 80:
                st.warning("⚠️ Низкая посещаемость. Старайтесь не пропускать занятия.")
