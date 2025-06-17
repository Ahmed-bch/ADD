import streamlit as st
import hashlib
import sqlite3
from datetime import datetime, timedelta
import secrets
import pandas as pd
import plotly.express as px
import os
import seaborn as sns
from io import BytesIO
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency
import numpy as np
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Advanced Data Analysis App", layout="wide")

# ============ CONFIGURATION ============

# Chemin de la base de données
DB_PATH = "app_users.db"

# ============ SYSTÈME D'AUTHENTIFICATION OPTIMISÉ ============

def diagnose_and_fix_database():
    """Diagnostique et répare la base de données"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Vérifier si la table existe
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='authorized_users';")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            st.sidebar.info("❌ Table 'authorized_users' n'existe pas - Création...")
            cursor.execute('''
                CREATE TABLE authorized_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
        
        # Supprimer tous les utilisateurs existants et les recréer
        cursor.execute("DELETE FROM authorized_users")
        
        # Recréer les utilisateurs par défaut
        default_users = [
            ("ahmed.djalil.2004@gmail․com", "ahmed26"),
            ("bouzamaamine8@gmail.com", "amine123"),
            ("a.a.boucherite@gmail.com","djalil26")
        ]
        
        for email, password in default_users:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            cursor.execute(
                "INSERT INTO authorized_users (email, password_hash, is_active) VALUES (?, ?, ?)",
                (email, password_hash, True)
            )
        
        conn.commit()
        
        # Vérification finale
        cursor.execute("SELECT email, is_active FROM authorized_users")
        users = cursor.fetchall()
        
        conn.close()
        return True, f"✅ {len(users)} utilisateur(s) créé(s)"
        
    except Exception as e:
        return False, f"❌ Erreur: {e}"

def init_database():
    """Initialise la base de données - VERSION CORRIGÉE"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Table des utilisateurs autorisés
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS authorized_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Table des sessions actives
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_sessions (
                session_id TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (email) REFERENCES authorized_users (email)
            )
        ''')
        
        # Vérifier si les utilisateurs existent ET s'ils sont corrects
        cursor.execute("SELECT COUNT(*) FROM authorized_users WHERE email = 'ahmed.djalil.2004@gmail․com'")
        admin_exists = cursor.fetchone()[0]
        
        # Forcer la recréation si admin n'existe pas
        if admin_exists == 0:
            # Supprimer tous les utilisateurs et recréer
            cursor.execute("DELETE FROM authorized_users")
            
            default_users = [
                ("ahmed.djalil.2004@gmail․com", "ahmed26"),
                ("bouzamaamine8@gmail.com", "amine123"),
                ("a.a.boucherite@gmail.com","djalil26")
            ]
            
            for email, password in default_users:
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                cursor.execute(
                    "INSERT OR REPLACE INTO authorized_users (email, password_hash, is_active) VALUES (?, ?, ?)",
                    (email, password_hash, True)
                )
        
        conn.commit()
        
    except Exception as e:
        st.sidebar.error(f"Erreur init DB: {e}")
        conn.rollback()
    finally:
        conn.close()

def hash_password(password):
    """Hash un mot de passe"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(email, password):
    """Vérifie les identifiants utilisateur - VERSION CORRIGÉE"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        password_hash = hash_password(password)
        
        # Vérifier d'abord si l'email existe
        cursor.execute("SELECT email, password_hash, is_active FROM authorized_users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            return False
        
        # Vérifier le mot de passe ET que l'utilisateur est actif
        if user[1] == password_hash and user[2]:
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"Erreur vérification: {e}")
        return False
    finally:
        conn.close()

def create_session(email):
    """Crée une nouvelle session pour un utilisateur"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Nettoyage des sessions expirées
    cursor.execute("DELETE FROM active_sessions WHERE expires_at < datetime('now')")
    
    # Vérifier les sessions actives pour cet utilisateur
    cursor.execute(
        "SELECT COUNT(*) FROM active_sessions WHERE email = ?", (email,)
    )
    
    if cursor.fetchone()[0] > 0:
        # Supprimer l'ancienne session et créer une nouvelle
        cursor.execute("DELETE FROM active_sessions WHERE email = ?", (email,))
    
    # Créer une nouvelle session
    session_id = secrets.token_urlsafe(16)
    expires_at = datetime.now() + timedelta(hours=2)
    
    cursor.execute(
        "INSERT INTO active_sessions (session_id, email, expires_at) VALUES (?, ?, ?)",
        (session_id, email, expires_at)
    )
    
    # Mettre à jour le last_login
    cursor.execute(
        "UPDATE authorized_users SET last_login = datetime('now') WHERE email = ?",
        (email,)
    )
    
    conn.commit()
    conn.close()
    return session_id, "Session créée"

def validate_session(session_id):
    """Valide une session active"""
    if not session_id:
        return False, None
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT email, expires_at FROM active_sessions WHERE session_id = ?",
        (session_id,)
    )
    
    result = cursor.fetchone()
    
    if result:
        email, expires_at = result
        expires_at = datetime.fromisoformat(expires_at)
        
        if expires_at > datetime.now():
            conn.close()
            return True, email
        else:
            # Session expirée
            cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
            conn.commit()
    
    conn.close()
    return False, None

def logout_session(session_id):
    """Supprime une session"""
    if session_id:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM active_sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

# ============ FONCTIONS DE SÉCURITÉ ============

def is_admin_user():
    """Vérifie si l'utilisateur actuel est administrateur"""
    session_id = st.session_state.get('session_id')
    is_valid, email = validate_session(session_id)
    return is_valid and email == 'ahmed.djalil.2004@gmail․com'

def secure_admin_sidebar():
    """Affiche la sidebar sécurisée pour les administrateurs uniquement"""
    if is_admin_user():
        st.sidebar.title("🔧 Outils de Debug Admin")
        st.sidebar.success("🔑 Accès Administrateur Confirmé")
        
        # Séparateur visuel
        st.sidebar.markdown("---")
        
        # Section Base de Données
        st.sidebar.subheader("💾 Base de Données")
        
        if st.sidebar.button("🚨 RÉPARER LA BASE", type="primary"):
            success, message = diagnose_and_fix_database()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
            st.rerun()
        
        if st.sidebar.button("🔄 Réinitialiser complètement", type="secondary"):
            # Confirmation de sécurité
            if st.sidebar.checkbox("⚠️ Confirmer la réinitialisation"):
                if os.path.exists(DB_PATH):
                    os.remove(DB_PATH)
                init_database()
                st.sidebar.success("✅ Base réinitialisée!")
                st.rerun()
        
        # Section Monitoring
        st.sidebar.subheader("📊 Monitoring")
        
        if st.sidebar.checkbox("📈 Statut de la base"):
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                
                # Statistiques utilisateurs
                cursor.execute("SELECT COUNT(*) FROM authorized_users")
                user_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM authorized_users WHERE email = 'admin@test.com'")
                admin_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM authorized_users WHERE is_active = 1")
                active_users = cursor.fetchone()[0]
                
                # Statistiques sessions
                cursor.execute("SELECT COUNT(*) FROM active_sessions")
                total_sessions = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM active_sessions WHERE datetime(expires_at) > datetime('now')")
                active_sessions = cursor.fetchone()[0]
                
                conn.close()
                
                # Affichage des métriques
                st.sidebar.metric("👥 Utilisateurs", user_count)
                st.sidebar.metric("✅ Utilisateurs actifs", active_users)
                st.sidebar.metric("🔐 Sessions totales", total_sessions)
                st.sidebar.metric("🟢 Sessions actives", active_sessions)
                st.sidebar.info(f"🔑 Admin: {'✅ Présent' if admin_count > 0 else '❌ Absent'}")
                
            except Exception as e:
                st.sidebar.error(f"❌ Erreur monitoring: {e}")
        
        # Section Actions rapides
        st.sidebar.subheader("⚡ Actions Rapides")
        
        if st.sidebar.button("🧹 Nettoyer sessions expirées"):
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM active_sessions WHERE expires_at < datetime('now')")
                deleted = cursor.rowcount
                conn.commit()
                conn.close()
                st.sidebar.success(f"✅ {deleted} session(s) nettoyée(s)")
            except Exception as e:
                st.sidebar.error(f"❌ Erreur: {e}")
    
    elif st.session_state.get('session_id'):
        # Utilisateur connecté mais pas admin
        st.sidebar.title("ℹ️ Informations")
        st.sidebar.info(f"👤 {st.session_state.get('user_email', 'Utilisateur')}")
        st.sidebar.warning("🚫 Outils d'administration\nréservés aux administrateurs")
    
    # Si pas connecté, ne rien afficher dans la sidebar

# ============ INTERFACES ============

def login_interface():
    """Interface de connexion simplifiée"""
    st.title("🔐 Connexion")
    
  
    
    with st.form("login_form", clear_on_submit=True):
        email = st.text_input("📧 Email")
        password = st.text_input("🔑 Mot de passe", type="password")
        
        if st.form_submit_button("Se connecter", type="primary"):
            if email and password:
                if verify_user(email, password):
                    session_id, message = create_session(email)
                    if session_id:
                        st.session_state['session_id'] = session_id
                        st.session_state['user_email'] = email
                        st.success("✅ Connexion réussie !")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("❌ Identifiants incorrects")
            else:
                st.warning("⚠️ Remplissez tous les champs")

def admin_panel():
    """Panneau d'administration intégré"""
    st.subheader("🔧 Administration")
    
    # Vérifier les droits admin
    if st.session_state.get('user_email') not in ['ahmed.djalil.2004@gmail․com']:
        st.error("🚫 Accès réservé aux administrateurs")
        return
    
    # Onglets d'administration
    admin_tab1, admin_tab2, admin_tab3 = st.tabs(["👥 Utilisateurs", "🔐 Sessions", "📊 Statistiques"])
    
    with admin_tab1:
        st.write("**Gestion des utilisateurs**")
        
        # Afficher les utilisateurs
        conn = sqlite3.connect(DB_PATH)
        try:
            df_users = pd.read_sql_query("""
                SELECT email, is_active, created_at, last_login 
                FROM authorized_users 
                ORDER BY created_at DESC
            """, conn)
            
            if not df_users.empty:
                st.dataframe(df_users, use_container_width=True)
            else:
                st.info("Aucun utilisateur trouvé")
            
            # Formulaire pour ajouter un utilisateur
            with st.form("add_user_form"):
                st.write("**➕ Ajouter un utilisateur**")
                col1, col2 = st.columns(2)
                
                with col1:
                    new_email = st.text_input("Email")
                with col2:
                    new_password = st.text_input("Mot de passe", type="password")
                
                if st.form_submit_button("Ajouter utilisateur"):
                    if new_email and new_password:
                        try:
                            cursor = conn.cursor()
                            password_hash = hashlib.sha256(new_password.encode()).hexdigest()
                            cursor.execute(
                                "INSERT INTO authorized_users (email, password_hash) VALUES (?, ?)",
                                (new_email, password_hash)
                            )
                            conn.commit()
                            st.success(f"✅ Utilisateur {new_email} ajouté")
                            st.rerun()
                        except sqlite3.IntegrityError:
                            st.error("❌ Cet utilisateur existe déjà")
                        except Exception as e:
                            st.error(f"❌ Erreur: {e}")
                    else:
                        st.warning("⚠️ Remplissez tous les champs")
        
        except Exception as e:
            st.error(f"Erreur lors de la récupération des utilisateurs: {e}")
        finally:
            conn.close()
    
    with admin_tab2:
        st.write("**Sessions actives**")
        
        conn = sqlite3.connect(DB_PATH)
        try:
            df_sessions = pd.read_sql_query("""
                SELECT 
                    email, 
                    substr(session_id, 1, 8) || '...' as session_id_short,
                    created_at, 
                    expires_at,
                    CASE 
                        WHEN datetime(expires_at) > datetime('now') THEN '🟢 Active'
                        ELSE '🔴 Expirée'
                    END as status
                FROM active_sessions 
                ORDER BY created_at DESC
            """, conn)
            
            if not df_sessions.empty:
                st.dataframe(df_sessions, use_container_width=True)
                
                if st.button("🧹 Nettoyer les sessions expirées"):
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM active_sessions WHERE expires_at < datetime('now')")
                    deleted = cursor.rowcount
                    conn.commit()
                    st.success(f"✅ {deleted} session(s) expirée(s) supprimée(s)")
                    st.rerun()
            else:
                st.info("Aucune session active")
        
        except Exception as e:
            st.error(f"Erreur lors de la récupération des sessions: {e}")
        finally:
            conn.close()
    
    with admin_tab3:
        st.write("**Statistiques de la base de données**")
        
        conn = sqlite3.connect(DB_PATH)
        try:
            # Métriques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                users_count = pd.read_sql_query("SELECT COUNT(*) as count FROM authorized_users", conn).iloc[0]['count']
                st.metric("👥 Utilisateurs", users_count)
            
            with col2:
                sessions_count = pd.read_sql_query("SELECT COUNT(*) as count FROM active_sessions", conn).iloc[0]['count']
                st.metric("🔐 Sessions totales", sessions_count)
            
            with col3:
                active_sessions = pd.read_sql_query("""
                    SELECT COUNT(*) as count 
                    FROM active_sessions 
                    WHERE datetime(expires_at) > datetime('now')
                """, conn).iloc[0]['count']
                st.metric("✅ Sessions actives", active_sessions)
        
        except Exception as e:
            st.error(f"Erreur lors de la récupération des statistiques: {e}")
        finally:
            conn.close()

def add():
    # Custom CSS to improve UI
    st.markdown("""
    <style>
        .main {
            padding: 1rem 2rem;
        }
        h1, h2, h3 {
            color: #1E88E5;
        }
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        .plot-container {
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 10px;
            border-radius: 5px;
            background-color: white;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("📊 Comprehensive Data Analysis Tool")
    st.markdown("Upload your Excel or CSV file and explore your data through various analyses and visualizations")

    # File uploader
    uploaded_file = st.file_uploader("Choose an Excel (.xlsx) or CSV (.csv) file", type=["xlsx", "csv"])

    # Main function
    def main():
        if uploaded_file is not None:
            # Read the data
            try:
                file_extension = uploaded_file.name.split('.')[-1].lower()
                if file_extension == "xlsx":
                    df = pd.read_excel(uploaded_file)
                elif file_extension == "csv":
                    df = pd.read_csv(uploaded_file)
                st.success("File successfully uploaded!")


                # Display tabs for different analyses
                tabs = st.tabs(["📋 Data Overview", "📊 Descriptive Statistics", "📈 Visualizations",
                    "🔥 Correlation Analysis","Clustering ", "📉 Advanced Analysis", "🔄 AFC","ACM","ACP"])

                with tabs[0]:
                    data_overview(df)

                with tabs[1]:
                    descriptive_statistics(df)

                with tabs[2]:
                    visualizations(df)

                with tabs[3]:
                    correlation_analysis(df)

                with tabs[4]:
                    k_means_analysis(df)
                    
                with tabs[5]:
                    advanced_analysis(df)
                
                with tabs[6]:
                    AFC_analysis(df)
                with tabs[7]:
                    ACM_analysis(df)
                with tabs[8]:
                    ACP_analysis(df)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Please upload an Excel file to begin analysis.")

            # Sample data option
            if st.button("Use Sample Data"):
                # Generate sample data
                np.random.seed(42)
                sample_data = pd.DataFrame({
                    'Age': np.random.normal(35, 10, 100).astype(int),
                    'Income': np.random.normal(50000, 15000, 100),
                    'Experience': np.random.normal(8, 5, 100),
                    'Satisfaction': np.random.randint(1, 6, 100),
                    'Department': np.random.choice(['Sales', 'Marketing', 'HR', 'IT', 'Finance'], 100),
                    'Performance': np.random.normal(7, 2, 100)
                })

                # Save to a BytesIO object
                output = BytesIO()
                sample_data.to_excel(output, index=False)
                output.seek(0)

                # Use the sample data
                df = sample_data
                st.success("Using sample data!")

                # Display tabs for different analyses
                tabs = st.tabs(["📋 Data Overview", "📊 Descriptive Statistics", "📈 Visualizations",
                                "🔥 Correlation Analysis","Clustering ", "📉 Advanced Analysis", "🔄 AFC","ACM","ACP"])

                with tabs[0]:
                    data_overview(df)

                with tabs[1]:
                    descriptive_statistics(df)

                with tabs[2]:
                    visualizations(df)

                with tabs[3]:
                    correlation_analysis(df)

                with tabs[4]:
                    advanced_analysis(df)
                with tabs[5]:
                    k_means_analysis(df)
                with tabs[6]:
                    AFC_analysis(df)
                with tabs[7]:
                    ACM_analysis(df)
                with tabs[8]:
                    ACP_analysis(df)

   
    def k_means_analysis(df):
        """
        Fonction complète d'analyse K-means et CAH avec interface Streamlit

        """

         # Analyse des clusters - Comparaison des deux méthodes
        st.subheader("📋 Analyse comparative des Clusters")

        # Sélection de l'algorithme à analyser
        analysis_algorithm = st.selectbox(
            "Analyser les clusters de:",
            ["K-means", "CAH", "Comparaison"],
            index=2,
            key="analysis_algo"
        )


        st.header("🎯 Analyse de Clustering : K-means vs CAH")
        
        # Sélection des colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ Vous avez besoin d'au moins 2 colonnes numériques pour effectuer l'analyse de clustering.")
            return
        
        # Interface utilisateur
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📊 Configuration de l'analyse")
            selected_features = st.multiselect(
                "Sélectionnez les variables pour le clustering:",
                numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))]
            )
        
        with col2:
            st.subheader("⚙️ Paramètres")
            max_clusters = st.slider("Nombre maximum de clusters à tester:", 2, 10, 8)
            normalize_data = st.checkbox("Normaliser les données", value=True)
            random_state = st.number_input("Graine aléatoire:", value=42, min_value=0)
            
            # Paramètres CAH
            st.subheader("🌳 Paramètres CAH")
            linkage_method = st.selectbox("Méthode de liaison:", 
                                        ['ward', 'complete', 'average', 'single'], 
                                        index=0)
        
        if len(selected_features) < 2:
            st.warning("⚠️ Veuillez sélectionner au moins 2 variables.")
            return
        
        # Bouton d'exécution
        if not st.button("🚀 Lancer l'analyse de clustering", type="primary"):
            st.info("👆 Cliquez sur le bouton ci-dessus pour lancer l'analyse.")
            return
        
        # Préparation des données
        X = df[selected_features].copy()
        
        # Gestion des valeurs manquantes
        if X.isnull().sum().sum() > 0:
            st.warning("⚠️ Valeurs manquantes détectées. Remplacement par la médiane.")
            X = X.fillna(X.median())
        
        # Normalisation
        if normalize_data:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=selected_features)
        else:
            X_scaled = X
        
        # Import des bibliothèques nécessaires
        from sklearn.cluster import KMeans, AgglomerativeClustering
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import pdist
        
        # Méthode du coude et score de silhouette pour K-means et CAH
        st.subheader("📈 Détermination du nombre optimal de clusters")
        
        # Créer deux colonnes pour K-means et CAH
        col_kmeans, col_cah = st.columns(2)
        
        with col_kmeans:
            st.write("### 🎯 K-means")
        with col_cah:
            st.write("### 🌳 CAH (Classification Ascendante Hiérarchique)")
        
        with st.spinner("Calcul en cours..."):
            # Métriques K-means
            kmeans_inertias = []
            kmeans_silhouette_scores = []
            kmeans_calinski_scores = []
            
            # Métriques CAH
            cah_silhouette_scores = []
            cah_calinski_scores = []
            
            k_range = range(2, max_clusters + 1)
            
            for k in k_range:
                # K-means
                kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                kmeans_labels = kmeans.fit_predict(X_scaled)
                
                kmeans_inertias.append(kmeans.inertia_)
                kmeans_silhouette_scores.append(silhouette_score(X_scaled, kmeans_labels))
                kmeans_calinski_scores.append(calinski_harabasz_score(X_scaled, kmeans_labels))
                
                # CAH
                cah = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
                cah_labels = cah.fit_predict(X_scaled)
                
                cah_silhouette_scores.append(silhouette_score(X_scaled, cah_labels))
                cah_calinski_scores.append(calinski_harabasz_score(X_scaled, cah_labels))
        
        # Visualisation des métriques - Comparaison K-means vs CAH
        st.subheader("📊 Comparaison des métriques K-means vs CAH")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_elbow = plt.figure(figsize=(10, 6))
            plt.plot(k_range, kmeans_inertias, 'bo-', markersize=8, linewidth=2, label='K-means (Inertie)')
            plt.title('Méthode du Coude (K-means uniquement)', fontsize=14, fontweight='bold')
            plt.xlabel('Nombre de clusters (k)')
            plt.ylabel('Inertie')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_elbow)
            plt.close()
        
        with col2:
            fig_silhouette = plt.figure(figsize=(10, 6))
            plt.plot(k_range, kmeans_silhouette_scores, 'bo-', markersize=8, linewidth=2, label='K-means')
            plt.plot(k_range, cah_silhouette_scores, 'ro-', markersize=8, linewidth=2, label='CAH')
            plt.title('Score de Silhouette', fontsize=14, fontweight='bold')
            plt.xlabel('Nombre de clusters (k)')
            plt.ylabel('Score de Silhouette')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_silhouette)
            plt.close()
        
        with col3:
            fig_calinski = plt.figure(figsize=(10, 6))
            plt.plot(k_range, kmeans_calinski_scores, 'bo-', markersize=8, linewidth=2, label='K-means')
            plt.plot(k_range, cah_calinski_scores, 'ro-', markersize=8, linewidth=2, label='CAH')
            plt.title('Score de Calinski-Harabasz', fontsize=14, fontweight='bold')
            plt.xlabel('Nombre de clusters (k)')
            plt.ylabel('Score CH')
            plt.legend()
            plt.grid(True, alpha=0.3)
            st.pyplot(fig_calinski)
            plt.close()
        
        # Dendrogramme pour CAH
        st.subheader("🌳 Dendrogramme CAH")
        
        # Calculer la matrice de liaison
        linkage_matrix = linkage(X_scaled, method=linkage_method)
        
        fig_dendro = plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, 
                truncate_mode='lastp',
                p=30,
                leaf_rotation=90,
                leaf_font_size=10,
                show_contracted=True)
        plt.title(f'Dendrogramme CAH (Méthode: {linkage_method})', fontsize=14, fontweight='bold')
        plt.xlabel('Échantillons ou (taille du cluster)')
        plt.ylabel('Distance')
        st.pyplot(fig_dendro)
        plt.close()
        
        # Recommandations automatiques
        optimal_k_kmeans_silhouette = k_range[np.argmax(kmeans_silhouette_scores)]
        optimal_k_kmeans_calinski = k_range[np.argmax(kmeans_calinski_scores)]
        optimal_k_cah_silhouette = k_range[np.argmax(cah_silhouette_scores)]
        optimal_k_cah_calinski = k_range[np.argmax(cah_calinski_scores)]
        
        st.info(f"🎯 **Recommandations automatiques:**\n"
                f"**K-means:**\n"
                f"- Meilleur score de Silhouette: k = {optimal_k_kmeans_silhouette}\n"
                f"- Meilleur score de Calinski-Harabasz: k = {optimal_k_kmeans_calinski}\n\n"
                f"**CAH:**\n"
                f"- Meilleur score de Silhouette: k = {optimal_k_cah_silhouette}\n"
                f"- Meilleur score de Calinski-Harabasz: k = {optimal_k_cah_calinski}")
        
        # Sélection du nombre de clusters
        st.subheader("🎯 Application des algorithmes de clustering")
        
        col1, col2 = st.columns(2)
        with col1:
            n_clusters_kmeans = st.selectbox(
                "Nombre de clusters pour K-means:",
                k_range,
                index=k_range.index(optimal_k_kmeans_silhouette),
                key="kmeans_clusters"
            )
        
        with col2:
            n_clusters_cah = st.selectbox(
                "Nombre de clusters pour CAH:",
                k_range,
                index=k_range.index(optimal_k_cah_silhouette),
                key="cah_clusters"
            )
        
        # Application des algorithmes finaux
        # K-means
        final_kmeans = KMeans(n_clusters=n_clusters_kmeans, random_state=random_state, n_init=10)
        kmeans_labels = final_kmeans.fit_predict(X_scaled)
        
        # CAH
        final_cah = AgglomerativeClustering(n_clusters=n_clusters_cah, linkage=linkage_method)
        cah_labels = final_cah.fit_predict(X_scaled)
        
        # Ajout des labels au dataframe original
        df_clustered = df.copy()
        df_clustered['Cluster_KMeans'] = kmeans_labels.astype(str)
        df_clustered['Cluster_CAH'] = cah_labels.astype(str)
        
        # Métriques de qualité
        kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
        kmeans_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
        cah_silhouette = silhouette_score(X_scaled, cah_labels)
        cah_calinski = calinski_harabasz_score(X_scaled, cah_labels)
        
        # Affichage des métriques
        st.subheader("📊 Comparaison des performances")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🎯 K-means")
            subcol1, subcol2, subcol3 = st.columns(3)
            with subcol1:
                st.metric("Silhouette", f"{kmeans_silhouette:.3f}")
            with subcol2:
                st.metric("Calinski-H", f"{kmeans_calinski:.1f}")
            with subcol3:
                st.metric("Inertie", f"{final_kmeans.inertia_:.1f}")
        
        with col2:
            st.write("### 🌳 CAH")
            subcol1, subcol2 = st.columns(2)
            with subcol1:
                st.metric("Silhouette", f"{cah_silhouette:.3f}")
            with subcol2:
                st.metric("Calinski-H", f"{cah_calinski:.1f}")
        
        # Déterminer le meilleur algorithme
        if kmeans_silhouette > cah_silhouette:
            st.success("🏆 **K-means** obtient un meilleur score de Silhouette!")
            best_algorithm = "K-means"
        elif cah_silhouette > kmeans_silhouette:
            st.success("🏆 **CAH** obtient un meilleur score de Silhouette!")
            best_algorithm = "CAH"
        else:
            st.info("🤝 Les deux algorithmes obtiennent des scores similaires!")
            best_algorithm = "Égalité"
        
        # Visualisations des clusters
        st.subheader("📊 Visualisation des Clusters - Comparaison")
        
        # Choix de l'algorithme à visualiser
        viz_algorithm = st.selectbox(
            "Choisir l'algorithme pour la visualisation:",
            ["K-means", "CAH", "Comparaison côte à côte"],
            index=2
        )
        
        if len(selected_features) >= 2:
            # Graphique en 2D
            col1, col2 = st.columns(2)
            
            with col1:
                var_x = st.selectbox("Variable X:", selected_features, key="x_var")
            with col2:
                var_y = st.selectbox("Variable Y:", selected_features, 
                                index=1 if len(selected_features) > 1 else 0, key="y_var")
            
            if viz_algorithm == "Comparaison côte à côte":
                col1, col2 = st.columns(2)
                
                with col1:
                    # K-means
                    fig_kmeans = px.scatter(
                        df_clustered, x=var_x, y=var_y, color='Cluster_KMeans',
                        title=f'K-means: {var_x} vs {var_y}',
                        color_discrete_sequence=px.colors.qualitative.Dark24
                    )
                    
                    # Ajout des centroïdes K-means
                    if normalize_data:
                        centroids_kmeans = scaler.inverse_transform(final_kmeans.cluster_centers_)
                    else:
                        centroids_kmeans = final_kmeans.cluster_centers_
                    
                    centroids_kmeans_df = pd.DataFrame(centroids_kmeans, columns=selected_features)
                    
                    fig_kmeans.add_scatter(
                        x=centroids_kmeans_df[var_x], y=centroids_kmeans_df[var_y],
                        mode='markers', marker=dict(symbol='x', size=15, color='black'),
                        name='Centroïdes', showlegend=True
                    )
                    
                    st.plotly_chart(fig_kmeans, use_container_width=True)
                
                with col2:
                    # CAH
                    fig_cah = px.scatter(
                        df_clustered, x=var_x, y=var_y, color='Cluster_CAH',
                        title=f'CAH: {var_x} vs {var_y}',
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    
                    st.plotly_chart(fig_cah, use_container_width=True)
            
            elif viz_algorithm == "K-means":
                fig_scatter = px.scatter(
                    df_clustered, x=var_x, y=var_y, color='Cluster_KMeans',
                    title=f'Clusters K-means: {var_x} vs {var_y}',
                    color_discrete_sequence=px.colors.qualitative.Dark24
                )
                
                # Ajout des centroïdes
                if normalize_data:
                    centroids = scaler.inverse_transform(final_kmeans.cluster_centers_)
                else:
                    centroids = final_kmeans.cluster_centers_
                
                centroids_df = pd.DataFrame(centroids, columns=selected_features)
                
                fig_scatter.add_scatter(
                    x=centroids_df[var_x], y=centroids_df[var_y],
                    mode='markers', marker=dict(symbol='x', size=15, color='black'),
                    name='Centroïdes', showlegend=True
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            else:  # CAH
                fig_scatter = px.scatter(
                    df_clustered, x=var_x, y=var_y, color='Cluster_CAH',
                    title=f'Clusters CAH: {var_x} vs {var_y}',
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Si plus de 2 variables, réduction dimensionnelle avec PCA
        if len(selected_features) > 2:
            st.subheader("🔍 Visualisation PCA (Réduction Dimensionnelle)")
            
            pca = PCA(n_components=2, random_state=random_state)
            X_pca = pca.fit_transform(X_scaled)
            
            if viz_algorithm == "Comparaison côte à côte":
                col1, col2 = st.columns(2)
                
                with col1:
                    pca_df_kmeans = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    pca_df_kmeans['Cluster'] = kmeans_labels.astype(str)

                    
                    fig_pca_kmeans = px.scatter(
                        pca_df_kmeans, x='PC1', y='PC2', color='Cluster',
                        title=f'K-means PCA - Variance: {pca.explained_variance_ratio_.sum():.2%}',
                        color_discrete_sequence=px.colors.qualitative.Dark24  # Nouvelle palette
                    )
                    
                    st.plotly_chart(fig_pca_kmeans, use_container_width=True)
                
                with col2:
                    pca_df_cah = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    pca_df_cah['Cluster'] = cah_labels.astype(str)
                    
                    fig_pca_cah = px.scatter(
                        pca_df_cah, x='PC1', y='PC2', color='Cluster',
                        title=f'CAH PCA - Variance: {pca.explained_variance_ratio_.sum():.2%}',
                        color_discrete_sequence=px.colors.qualitative.Set2  # Nouvelle palette
                    )
                    
                    st.plotly_chart(fig_pca_cah, use_container_width=True)
            
            else:
                pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                if viz_algorithm == "K-means":
                    pca_df['Cluster'] = kmeans_labels
                    color_palette = px.colors.qualitative.Dark24  # Nouvelle palette
                else:
                    pca_df['Cluster'] = cah_labels
                    color_palette = px.colors.qualitative.Prism  # Nouvelle palette
                
                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', color='Cluster',
                    title=f'Clusters {viz_algorithm} (PCA) - Variance expliquée: {pca.explained_variance_ratio_.sum():.2%}',
                    color_discrete_sequence=color_palette
                )
                
                st.plotly_chart(fig_pca, use_container_width=True)
            
            # Information sur les composantes principales
            st.write("**Contribution des variables aux composantes principales:**")
            components_df = pd.DataFrame(
                pca.components_.T,
                columns=['PC1', 'PC2'],
                index=selected_features
            )
            st.dataframe(components_df.round(3))
        
        
        if analysis_algorithm == "K-means":
            # Analyse K-means uniquement
            for i in range(n_clusters_kmeans):
                with st.expander(f"🎯 K-means - Cluster {i} (n = {len(df_clustered[df_clustered['Cluster_KMeans'] == str(i)])})"):
                    cluster_data = df_clustered[df_clustered['Cluster_KMeans'] == str(i)]
                    
                    # Statistiques descriptives
                    st.write("**Moyennes des variables:**")
                    means = cluster_data[selected_features].mean()
                    st.write(means.round(3))
                    
                    # Comparaison avec la moyenne globale
                    st.write("**Écart par rapport à la moyenne globale:**")
                    global_means = df[selected_features].mean()
                    differences = ((means - global_means) / global_means * 100).round(1)
                    st.write(f"{differences}%")
                    
                    # Variables catégorielles si disponibles
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if categorical_cols:
                        st.write("**Distribution des variables catégorielles:**")
                        for cat_col in categorical_cols[:3]:
                            mode_val = cluster_data[cat_col].mode()
                            if len(mode_val) > 0:
                                st.write(f"- {cat_col}: {mode_val.iloc[0]} ({cluster_data[cat_col].value_counts().iloc[0]} occurrences)")
        
        elif analysis_algorithm == "CAH":
            # Analyse CAH uniquement
            for i in range(n_clusters_cah):
                with st.expander(f"🌳 CAH - Cluster {i} (n = {len(df_clustered[df_clustered['Cluster_CAH'] == str(i)])})"):
                    cluster_data = df_clustered[df_clustered['Cluster_CAH'] == str(i)]
                    
                    # Statistiques descriptives
                    st.write("**Moyennes des variables:**")
                    means = cluster_data[selected_features].mean()
                    st.write(means.round(3))
                    
                    # Comparaison avec la moyenne globale
                    st.write("**Écart par rapport à la moyenne globale:**")
                    global_means = df[selected_features].mean()
                    differences = ((means - global_means) / global_means * 100).round(1)
                    st.write(f"{differences}%")
                    
                    # Variables catégorielles si disponibles
                    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                    if categorical_cols:
                        st.write("**Distribution des variables catégorielles:**")
                        for cat_col in categorical_cols[:3]:
                            mode_val = cluster_data[cat_col].mode()
                            if len(mode_val) > 0:
                                st.write(f"- {cat_col}: {mode_val.iloc[0]} ({cluster_data[cat_col].value_counts().iloc[0]} occurrences)")
        
        else:
            # Comparaison côte à côte
            max_clusters_comp = max(n_clusters_kmeans, n_clusters_cah)
            
            for i in range(max_clusters_comp):
                col1, col2 = st.columns(2)
                
                with col1:
                    if i < n_clusters_kmeans:
                        with st.expander(f"🎯 K-means - Cluster {i}"):
                            cluster_data_kmeans = df_clustered[df_clustered['Cluster_KMeans'] == str(i)]
                            st.write(f"**Taille:** {len(cluster_data_kmeans)} observations")
                            
                            means_kmeans = cluster_data_kmeans[selected_features].mean()
                            st.write("**Moyennes:**")
                            st.write(means_kmeans.round(3))
                            
                            global_means = df[selected_features].mean()
                            differences_kmeans = ((means_kmeans - global_means) / global_means * 100).round(1)
                            st.write("**Écarts (%):**")
                            st.write(differences_kmeans)
                    else:
                        st.write("—")
                
                with col2:
                    if i < n_clusters_cah:
                        with st.expander(f"🌳 CAH - Cluster {i}"):
                            cluster_data_cah = df_clustered[df_clustered['Cluster_CAH'] == str(i)]
                            st.write(f"**Taille:** {len(cluster_data_cah)} observations")
                            
                            means_cah = cluster_data_cah[selected_features].mean()
                            st.write("**Moyennes:**")
                            st.write(means_cah.round(3))
                            
                            global_means = df[selected_features].mean()
                            differences_cah = ((means_cah - global_means) / global_means * 100).round(1)
                            st.write("**Écarts (%):**")
                            st.write(differences_cah)
                    else:
                        st.write("—")
        
        # Matrice de concordance entre les deux méthodes
        if n_clusters_kmeans == n_clusters_cah:
            st.subheader("🔄 Matrice de Concordance entre K-means et CAH")
            
            # Créer une matrice de confusion
            concordance_matrix = pd.crosstab(
                df_clustered['Cluster_KMeans'], 
                df_clustered['Cluster_CAH'], 
                margins=True
            )
            
            st.write("**Tableau de concordance:**")
            st.dataframe(concordance_matrix)
            
            # Calcul de l'accord entre les deux méthodes
            agreement = np.sum(np.diag(concordance_matrix.values[:-1, :-1])) / len(df_clustered)
            st.write(f"**Taux d'accord entre les deux méthodes:** {agreement:.2%}")
            
            # Heatmap de concordance
            fig_concordance = plt.figure(figsize=(8, 6))
            sns.heatmap(concordance_matrix.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues')
            plt.title('Matrice de Concordance K-means vs CAH')
            plt.xlabel('Clusters CAH')
            plt.ylabel('Clusters K-means')
            st.pyplot(fig_concordance)
            plt.close()
        
        # Centroïdes et Distances
        st.subheader("🎯 Centroïdes et Analyse des Distances")
        
        # Affichage des centroïdes pour K-means
        if normalize_data:
            centroids_display_kmeans = scaler.inverse_transform(final_kmeans.cluster_centers_)
        else:
            centroids_display_kmeans = final_kmeans.cluster_centers_
        
        centroids_df_kmeans = pd.DataFrame(
            centroids_display_kmeans,
            columns=selected_features,
            index=[f'K-means Cluster {i}' for i in range(n_clusters_kmeans)]
        )
        
        # Calcul des centroïdes pour CAH
        centroids_cah = []
        for i in range(n_clusters_cah):
            cluster_data = df_clustered[df_clustered['Cluster_CAH'] == str(i)]
            centroid = cluster_data[selected_features].mean().values
            centroids_cah.append(centroid)
        
        centroids_df_cah = pd.DataFrame(
            centroids_cah,
            columns=selected_features,
            index=[f'CAH Cluster {i}' for i in range(n_clusters_cah)]
        )
        
        # Affichage des centroïdes
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🎯 Centroïdes K-means:**")
            st.dataframe(centroids_df_kmeans.round(3))
        
        with col2:
            st.write("**🌳 Centroïdes CAH:**")
            st.dataframe(centroids_df_cah.round(3))
        
        # Heatmaps des centroïdes
        if len(selected_features) > 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Heatmap Centroïdes K-means:**")
                fig_heatmap_kmeans = plt.figure(figsize=(10, 6))
                sns.heatmap(centroids_df_kmeans.T, annot=True, cmap='viridis', center=0, fmt='.2f')
                plt.title('Heatmap des Centroïdes K-means')
                plt.tight_layout()
                st.pyplot(fig_heatmap_kmeans)
                plt.close()
            
            with col2:
                st.write("**Heatmap Centroïdes CAH:**")
                fig_heatmap_cah = plt.figure(figsize=(10, 6))
                sns.heatmap(centroids_df_cah.T, annot=True, cmap='plasma', center=0, fmt='.2f')
                plt.title('Heatmap des Centroïdes CAH')
                plt.tight_layout()
                st.pyplot(fig_heatmap_cah)
                plt.close()
        
        # Export des résultats
        st.subheader("💾 Export des Résultats")
        
        # Choix du format d'export
        export_choice = st.selectbox(
            "Choisir les données à exporter:",
            ["K-means uniquement", "CAH uniquement", "Les deux algorithmes"],
            index=2
        )
        
        if export_choice == "K-means uniquement":
            export_df = df.copy()
            export_df['Cluster'] = kmeans_labels
            filename = f"data_kmeans_clusters_{n_clusters_kmeans}.csv"
        elif export_choice == "CAH uniquement":
            export_df = df.copy()
            export_df['Cluster'] = cah_labels
            filename = f"data_cah_clusters_{n_clusters_cah}.csv"
        else:
            export_df = df_clustered
            filename = f"data_clustering_comparison.csv"
        
        csv_results = export_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données avec clusters",
            data=csv_results,
            file_name=filename,
            mime="text/csv"
        )
        
        # Résumé des insights
        st.subheader("💡 Insights et Recommandations")
        
        insights = []
        
        # Comparaison des performances
        if kmeans_silhouette > cah_silhouette:
            insights.append(f"🏆 **K-means** montre de meilleures performances (Silhouette: {kmeans_silhouette:.3f} vs {cah_silhouette:.3f})")
        elif cah_silhouette > kmeans_silhouette:
            insights.append(f"🏆 **CAH** montre de meilleures performances (Silhouette: {cah_silhouette:.3f} vs {kmeans_silhouette:.3f})")
        else:
            insights.append("🤝 Les deux algorithmes montrent des performances similaires")
        
        # Taille des clusters K-means
        cluster_sizes_kmeans = df_clustered['Cluster_KMeans'].value_counts().sort_index()
        largest_cluster_kmeans = cluster_sizes_kmeans.idxmax()
        smallest_cluster_kmeans = cluster_sizes_kmeans.idxmin()
        
        insights.append(f"📊 **K-means:** Cluster le plus grand = {largest_cluster_kmeans} ({cluster_sizes_kmeans[largest_cluster_kmeans]} obs.), "
                    f"le plus petit = {smallest_cluster_kmeans} ({cluster_sizes_kmeans[smallest_cluster_kmeans]} obs.)")
        
        # Taille des clusters CAH
        cluster_sizes_cah = df_clustered['Cluster_CAH'].value_counts().sort_index()
        largest_cluster_cah = cluster_sizes_cah.idxmax()
        smallest_cluster_cah = cluster_sizes_cah.idxmin()
        
        insights.append(f"🌳 **CAH:** Cluster le plus grand = {largest_cluster_cah} ({cluster_sizes_cah[largest_cluster_cah]} obs.), "
                    f"le plus petit = {smallest_cluster_cah} ({cluster_sizes_cah[smallest_cluster_cah]} obs.)")
        
        # Variables les plus discriminantes
        if len(selected_features) > 1:
            # Pour K-means
            kmeans_cluster_means = df_clustered.groupby('Cluster_KMeans')[selected_features].mean()
            kmeans_feature_variance = kmeans_cluster_means.var()
            most_discriminant_kmeans = kmeans_feature_variance.idxmax()
            
            # Pour CAH
            cah_cluster_means = df_clustered.groupby('Cluster_CAH')[selected_features].mean()
            cah_feature_variance = cah_cluster_means.var()
            most_discriminant_cah = cah_feature_variance.idxmax()
            
            insights.append(f"🔍 Variable la plus discriminante pour **K-means:** '{most_discriminant_kmeans}'")
            insights.append(f"🔍 Variable la plus discriminante pour **CAH:** '{most_discriminant_cah}'")
        
        # Recommandation finale
        if best_algorithm == "K-means":
            insights.append("🎯 **Recommandation:** Utiliser K-means pour cette analyse")
        elif best_algorithm == "CAH":
            insights.append("🎯 **Recommandation:** Utiliser CAH pour cette analyse")
        else:
            insights.append("🎯 **Recommandation:** Les deux méthodes sont équivalentes, choisir selon le contexte")
        
        # Accord entre méthodes (si même nombre de clusters)
        if n_clusters_kmeans == n_clusters_cah:
            agreement = np.sum(np.diag(pd.crosstab(df_clustered['Cluster_KMeans'], df_clustered['Cluster_CAH']).values)) / len(df_clustered)
            if agreement > 0.7:
                insights.append(f"✅ Bon accord entre les deux méthodes ({agreement:.1%}) - résultats cohérents")
            elif agreement > 0.5:
                insights.append(f"⚠️ Accord modéré entre les deux méthodes ({agreement:.1%}) - vérifier l'interprétation")
            else:
                insights.append(f"❌ Faible accord entre les deux méthodes ({agreement:.1%}) - résultats divergents")
        
        for insight in insights:
            st.write(insight)

    def ACP_analysis(df):
        """
        Fonction d'Analyse en Composantes Principales avec biplot amélioré et tests statistiques
        Version améliorée avec validation statistique complète
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        # Imports essentiels
        import streamlit as st
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA as SklearnPCA
        from scipy import stats
        from scipy.stats import pearsonr, bartlett, levene
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Configuration du style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Vérifier si la bibliothèque fanalysis est disponible
        USE_FANALYSIS = False
        try:
            from fanalysis.pca import PCA
            USE_FANALYSIS = True
        except ImportError:
            st.warning("⚠️ La bibliothèque 'fanalysis' n'est pas disponible. Utilisation de sklearn avec fonctionnalités étendues.")
        
        st.header("🔍 Analyse en Composantes Principales (ACP) Avancée")
        st.markdown("Analysez les relations entre variables quantitatives avec validation statistique complète")
        
        if df.empty:
            st.error("❌ Aucune donnée disponible")
            return
        
        # ==================== SECTION CONFIGURATION ====================
        st.markdown("## ⚙️ Configuration de l'analyse")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Paramètres généraux")
            
            # Gestion des valeurs manquantes
            missing_method = st.selectbox(
                "Gestion des valeurs manquantes:",
                ["Supprimer les lignes", "Remplacer par moyenne", "Remplacer par médiane", 
                "Interpolation linéaire", "Ne rien faire"],
                key="acp_missing_method",
                help="Méthode pour traiter les valeurs manquantes"
            )
            
            # Standardisation
            standardize = st.checkbox(
                "Standardiser les données",
                value=True,
                key="acp_standardize",
                help="Recommandé quand les variables ont des unités différentes"
            )
            
            # Tests statistiques
            run_statistical_tests = st.checkbox(
                "Exécuter les tests statistiques",
                value=True,
                key="acp_run_tests",
                help="Tests de normalité, homoscédasticité, etc."
            )
            
            # Index personnalisé
            use_custom_index = st.checkbox("Utiliser une colonne comme index", key="acp_use_custom_index")
            if use_custom_index:
                index_col = st.selectbox(
                    "Colonne pour l'index:",
                    options=df.columns.tolist(),
                    key="acp_index_col",
                    help="Cette colonne sera utilisée comme identifiant"
                )
        
        with col2:
            st.subheader("📊 Aperçu des données")
            st.write("**Premières lignes:**")
            st.dataframe(df.head())
            
            # Informations détaillées sur les données
            info_data = {
                'Type': df.dtypes.astype(str),
                'Valeurs manquantes': df.isnull().sum(),
                '% manquantes': (df.isnull().sum() / len(df) * 100).round(2),
                'Valeurs uniques': df.nunique(),
                'Min': df.select_dtypes(include=[np.number]).min(),
                'Max': df.select_dtypes(include=[np.number]).max()
            }
            info_df = pd.DataFrame(info_data)
            with st.expander("Informations détaillées"):
                st.dataframe(info_df)
        
        # ==================== PRÉPROCESSING ====================
        df_processed = df.copy()
        
        # Appliquer l'index personnalisé
        if use_custom_index and 'index_col' in locals():
            df_processed.set_index(index_col, inplace=True)
            st.info(f"✅ Index défini sur la colonne: {index_col}")
        
        # Traitement des valeurs manquantes
        missing_before = df_processed.isnull().sum().sum()
        
        if missing_method == "Supprimer les lignes":
            df_processed = df_processed.dropna()
            st.info(f"📉 Lignes supprimées: {len(df) - len(df_processed)}")
        elif missing_method == "Remplacer par moyenne":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif missing_method == "Remplacer par médiane":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        elif missing_method == "Interpolation linéaire":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].interpolate(method='linear')
        
        missing_after = df_processed.isnull().sum().sum()
        if missing_before > 0:
            st.info(f"📊 Valeurs manquantes traitées: {missing_before} → {missing_after}")
        
        # ==================== SÉLECTION DES VARIABLES ====================
        st.markdown("---")
        st.subheader("📋 Sélection et validation des variables")
        
        # Détecter automatiquement les variables numériques
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.error("❌ L'ACP nécessite au moins 2 variables numériques")
            return
        
        # Interface pour sélection des variables
        col1, col2 = st.columns(2)
        
        with col1:
            vars_selected = st.multiselect(
                "Variables pour l'analyse:",
                options=numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))],
                key="acp_vars_selected",
                help="Sélectionnez les variables numériques (max 20 recommandé)"
            )
            
            if len(vars_selected) > 20:
                st.warning("⚠️ Plus de 20 variables peut ralentir l'analyse")
        
        with col2:
            if vars_selected:
                st.write("**Variables sélectionnées:**")
                for var in vars_selected[:10]:  # Limiter l'affichage
                    min_val = df_processed[var].min()
                    max_val = df_processed[var].max()
                    std_val = df_processed[var].std()
                    st.write(f"• {var}: [{min_val:.2f}, {max_val:.2f}] (σ={std_val:.2f})")
                if len(vars_selected) > 10:
                    st.write(f"... et {len(vars_selected) - 10} autres variables")
        
        if len(vars_selected) < 2:
            st.warning("⚠️ Veuillez sélectionner au moins 2 variables")
            return
        
        # Créer le dataset final
        X = df_processed[vars_selected].copy()
        
        # ==================== TESTS STATISTIQUES PRÉLIMINAIRES ====================
        if run_statistical_tests:
            st.markdown("---")
            st.subheader("🧪 Tests statistiques préliminaires")
            
            # Test de normalité (Shapiro-Wilk pour échantillons < 5000, sinon Kolmogorov-Smirnov)
            normality_results = {}
            for col in X.columns:
                data = X[col].dropna()
                if len(data) < 5000:
                    stat, p_value = stats.shapiro(data)
                    test_name = "Shapiro-Wilk"
                else:
                    stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
                    test_name = "Kolmogorov-Smirnov"
                
                normality_results[col] = {
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > 0.05
                }
            
            # Affichage des résultats de normalité
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Tests de normalité:**")
                normal_count = sum(1 for r in normality_results.values() if r['is_normal'])
                total_count = len(normality_results)
                st.metric("Variables normales", f"{normal_count}/{total_count}")
                
                # Graphique de distribution
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                axes = axes.ravel()
                
                for i, col in enumerate(X.columns[:4]):  # Montrer les 4 premières
                    if i < len(axes):
                        axes[i].hist(X[col], bins=30, alpha=0.7, density=True)
                        axes[i].set_title(f'{col}\n(Normal: {normality_results[col]["is_normal"]})')
                        
                        # Ajouter courbe normale théorique
                        x_norm = np.linspace(X[col].min(), X[col].max(), 100)
                        y_norm = stats.norm.pdf(x_norm, X[col].mean(), X[col].std())
                        axes[i].plot(x_norm, y_norm, 'r-', alpha=0.8)
                
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.write("**Détails des tests:**")
                normality_df = pd.DataFrame(normality_results).T
                normality_df = normality_df.round(4)
                st.dataframe(normality_df)
            
            # Test d'homoscédasticité (Bartlett et Levene)
            if len(X.columns) > 2:
                st.write("**Tests d'homoscédasticité:**")
                try:
                    # Test de Bartlett (assume normalité)
                    bartlett_stat, bartlett_p = bartlett(*[X[col].dropna() for col in X.columns])
                    
                    # Test de Levene (plus robuste)
                    levene_stat, levene_p = levene(*[X[col].dropna() for col in X.columns])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Test de Bartlett", f"p = {bartlett_p:.4f}")
                        if bartlett_p > 0.05:
                            st.success("✅ Variances homogènes (Bartlett)")
                        else:
                            st.warning("⚠️ Variances hétérogènes (Bartlett)")
                    
                    with col2:
                        st.metric("Test de Levene", f"p = {levene_p:.4f}")
                        if levene_p > 0.05:
                            st.success("✅ Variances homogènes (Levene)")
                        else:
                            st.warning("⚠️ Variances hétérogènes (Levene)")
                            
                except Exception as e:
                    st.warning(f"Impossible d'effectuer les tests d'homoscédasticité: {e}")
        
        # ==================== STANDARDISATION ====================
        if standardize:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            X = X_scaled
            st.success("✅ Données standardisées (moyenne=0, écart-type=1)")
        
        # ==================== ANALYSE DES CORRÉLATIONS ====================
        st.markdown("---")
        st.subheader("📈 Analyse des corrélations")
        
        # Calcul de la matrice de corrélation
        correlation_matrix = X.corr()
        
        # Visualisation interactive avec Plotly
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            title="Matrice de corrélation interactive",
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=600
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Métriques de corrélation
        mean_correlation = correlation_matrix.abs().mean().mean()
        strong_correlations = (correlation_matrix.abs() > 0.7).sum().sum() - len(vars_selected)
        very_strong_correlations = (correlation_matrix.abs() > 0.9).sum().sum() - len(vars_selected)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Corrélation moyenne", f"{mean_correlation:.3f}")
        with col2:
            st.metric("Corrélations fortes (>0.7)", strong_correlations)
        with col3:
            st.metric("Corrélations très fortes (>0.9)", very_strong_correlations)
        
        # Test d'adéquation pour l'ACP avec critères multiples
        st.markdown("---")
        st.subheader("🎯 Adéquation des données pour l'ACP")
        
        # Critère de corrélation
        if mean_correlation > 0.6:
            corr_adequacy = "Excellente"
            corr_color = "🟢"
        elif mean_correlation > 0.4:
            corr_adequacy = "Bonne"
            corr_color = "🟡"
        elif mean_correlation > 0.2:
            corr_adequacy = "Moyenne"
            corr_color = "🟠"
        else:
            corr_adequacy = "Faible"
            corr_color = "🔴"
        
        # Test KMO (Kaiser-Meyer-Olkin) approximatif
        def calculate_kmo_approx(corr_matrix):
            """Calcul approximatif du KMO"""
            corr_inv = np.linalg.pinv(corr_matrix)
            partial_corr = np.zeros_like(corr_matrix)
            
            for i in range(len(corr_matrix)):
                for j in range(len(corr_matrix)):
                    if i != j:
                        partial_corr[i, j] = -corr_inv[i, j] / np.sqrt(corr_inv[i, i] * corr_inv[j, j])
            
            sum_corr_sq = np.sum(corr_matrix**2) - np.trace(corr_matrix**2)
            sum_partial_sq = np.sum(partial_corr**2)
            
            kmo = sum_corr_sq / (sum_corr_sq + sum_partial_sq)
            return kmo
        
        try:
            kmo_value = calculate_kmo_approx(correlation_matrix.values)
            if kmo_value > 0.8:
                kmo_adequacy = "Excellent"
                kmo_color = "🟢"
            elif kmo_value > 0.7:
                kmo_adequacy = "Bon"
                kmo_color = "🟡"
            elif kmo_value > 0.6:
                kmo_adequacy = "Moyen"
                kmo_color = "🟠"
            else:
                kmo_adequacy = "Insuffisant"
                kmo_color = "🔴"
        except:
            kmo_value = None
            kmo_adequacy = "Non calculable"
            kmo_color = "⚪"
        
        # Affichage des critères d'adéquation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Corrélation moyenne:** {corr_color} {corr_adequacy}")
            st.write(f"Valeur: {mean_correlation:.3f}")
        
        with col2:
            if kmo_value:
                st.markdown(f"**Test KMO (approx.):** {kmo_color} {kmo_adequacy}")
                st.write(f"Valeur: {kmo_value:.3f}")
            else:
                st.markdown("**Test KMO:** ⚪ Non calculable")
        
        with col3:
            # Déterminant de la matrice de corrélation
            det_corr = np.linalg.det(correlation_matrix)
            if det_corr < 0.00001:
                det_adequacy = "Multicolinéarité détectée"
                det_color = "🔴"
            else:
                det_adequacy = "Acceptable"
                det_color = "🟢"
            
            st.markdown(f"**Déterminant:** {det_color} {det_adequacy}")
            st.write(f"Valeur: {det_corr:.2e}")
        
        # ==================== PARAMÈTRES ACP ====================
        st.markdown("---")
        st.subheader("⚙️ Paramètres de l'ACP")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            n_components = st.slider(
                "Nombre de composantes", 
                2, 
                min(len(vars_selected), len(X)), 
                min( len(X), len(vars_selected)), 
                key="acp_n_components"
            )
        with col2:
            fig_size = st.slider("Taille des graphiques", 6, 15, 10, key="acp_fig_size")
        with col3:
            use_plotly = st.checkbox("Graphiques interactifs (Plotly)", value=True, key="acp_use_plotly")
        
        # ==================== EXÉCUTION DE L'ACP ====================
        st.markdown("---")
        
        if st.button("🚀 Lancer l'Analyse ACP Complète", type="primary", key="acp_launch_button"):
            
            # Données finales
            st.subheader("✅ Données préparées pour l'ACP")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Aperçu final:**")
                st.dataframe(X.head())
            
            with col2:
                st.write("**Statistiques descriptives:**")
                desc_stats = X.describe().round(3)
                st.dataframe(desc_stats)
            
            # Exécution de l'ACP
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Choix de la méthode ACP
                if USE_FANALYSIS:
                    status_text.text("Utilisation de fanalysis PCA...")
                    progress_bar.progress(20)
                    
                    acp = PCA(
                        row_labels=X.index.values, 
                        col_labels=X.columns.values, 
                        n_components=n_components
                    )
                    acp.fit(X.values)
                    
                    eigenvalues = acp.eig_[1]
                    variance_explained = (eigenvalues / eigenvalues.sum()) * 100
                    cumulative_variance = np.cumsum(variance_explained)
                    
                else:
                    status_text.text("Utilisation de sklearn PCA...")
                    progress_bar.progress(20)
                    
                    # ACP avec sklearn
                    pca_sklearn = SklearnPCA(n_components=n_components)
                    X_transformed = pca_sklearn.fit_transform(X)
                    
                    eigenvalues = pca_sklearn.explained_variance_
                    variance_explained = pca_sklearn.explained_variance_ratio_ * 100
                    cumulative_variance = np.cumsum(variance_explained)
                    
                    # Créer des objets compatibles pour la suite
                    components = pca_sklearn.components_
                    
                progress_bar.progress(40)
                
                # ==================== RÉSULTATS PRINCIPAUX ====================
                st.markdown("---")
                st.subheader("🎯 Résultats de l'ACP")
                
                # Valeurs propres et variance
                st.subheader("📊 Valeurs propres et variance expliquée")
                
                # Graphiques des valeurs propres
                col1, col2 = st.columns(2)
                
                with col1:
                    if use_plotly:
                        fig_eigen = go.Figure()
                        fig_eigen.add_trace(go.Bar(
                            x=[f'PC{i+1}' for i in range(len(eigenvalues))],
                            y=eigenvalues,
                            name='Valeurs propres',
                            marker_color='lightblue'
                        ))
                        fig_eigen.add_hline(y=1, line_dash="dash", line_color="red", 
                                        annotation_text="Critère de Kaiser")
                        fig_eigen.update_layout(title="Valeurs propres", height=400)
                        st.plotly_chart(fig_eigen, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        bars = ax.bar(range(len(eigenvalues)), eigenvalues, alpha=0.7)
                        ax.axhline(y=1, color='red', linestyle='--', label='Critère de Kaiser')
                        ax.set_xlabel('Composantes')
                        ax.set_ylabel('Valeurs propres')
                        ax.set_title('Valeurs propres')
                        ax.set_xticks(range(len(eigenvalues)))
                        ax.set_xticklabels([f'PC{i+1}' for i in range(len(eigenvalues))])
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                
                with col2:
                    if use_plotly:
                        fig_cum = go.Figure()
                        fig_cum.add_trace(go.Scatter(
                            x=[f'PC{i+1}' for i in range(len(cumulative_variance))],
                            y=cumulative_variance,
                            mode='lines+markers',
                            name='Variance cumulée',
                            line=dict(color='orange', width=3)
                        ))
                        fig_cum.add_hline(y=80, line_dash="dash", line_color="green", 
                                        annotation_text="80% de variance")
                        fig_cum.update_layout(title="Variance cumulée (%)", height=400)
                        st.plotly_chart(fig_cum, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(range(len(cumulative_variance)), cumulative_variance, 'o-', linewidth=2)
                        ax.axhline(y=80, color='green', linestyle='--', label='80% variance')
                        ax.set_xlabel('Composantes')
                        ax.set_ylabel('Variance cumulée (%)')
                        ax.set_title('Variance cumulée')
                        ax.set_xticks(range(len(cumulative_variance)))
                        ax.set_xticklabels([f'PC{i+1}' for i in range(len(cumulative_variance))])
                        ax.legend()
                        plt.tight_layout()
                        st.pyplot(fig)
                
                progress_bar.progress(60)
                
                # Tableau détaillé des valeurs propres
                eigenvalues_df = pd.DataFrame({
                    'Composante': [f'PC{i+1}' for i in range(len(eigenvalues))],
                    'Valeur propre': eigenvalues.round(4),
                    'Variance (%)': variance_explained.round(2),
                    'Variance cumulée (%)': cumulative_variance.round(2)
                })
                
                st.write("**Tableau des valeurs propres:**")
                st.dataframe(eigenvalues_df, use_container_width=True)
                
                # Critères de sélection des composantes
                kaiser_threshold = 1.0
                significant_kaiser = sum(eigenvalues > kaiser_threshold)
                variance_80 = sum(cumulative_variance <= 80)
                variance_90 = sum(cumulative_variance <= 90)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Critère de Kaiser (>1)", f"{significant_kaiser} composantes")
                with col2:
                    st.metric("80% de variance", f"{variance_80} composantes")
                with col3:
                    st.metric("90% de variance", f"{variance_90} composantes")
                
                progress_bar.progress(80)
                
                # ==================== VISUALISATIONS INTERACTIVES ====================
                st.markdown("---")
                st.subheader("📈 Visualisations interactives")
                
                # Sélection des axes
                col1, col2 = st.columns(2)
                with col1:
                    axis_x = st.selectbox("Axe X", [f"PC{i+1}" for i in range(n_components)], 
                                        index=0, key="acp_axis_x_viz")
                    num_x_axis = int(axis_x[2:])
                with col2:
                    axis_y = st.selectbox("Axe Y", [f"PC{i+1}" for i in range(n_components)], 
                                        index=1, key="acp_axis_y_viz")
                    num_y_axis = int(axis_y[2:])
                
                # Graphiques principaux
                tab1, tab2, tab3, tab4 = st.tabs(["👥 Individus", "📊 Variables", "🎭 Biplot Standard", "🎯 Biplot Amélioré"])
                
                with tab1:
                    st.write("**Projection des individus**")
                    fig_rows = acp.mapping_row(num_x_axis=num_x_axis, num_y_axis=num_y_axis, 
                                            figsize=(fig_size, fig_size))
                    plt.title(f"Individus - Plan PC{num_x_axis}-PC{num_y_axis}", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig_rows)
                
                with tab2:
                    st.write("**Cercle des corrélations**")
                    fig_cols = acp.correlation_circle(num_x_axis=num_x_axis, num_y_axis=num_y_axis,
                                            figsize=(fig_size, fig_size))
                    plt.title(f"Cercle des corrélations - Plan PC{num_x_axis}-PC{num_y_axis}", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig_cols)
                
                with tab3:
                    st.write("**Biplot Standard (Individus + Variables)**")
                    fig_biplot = acp.mapping(num_x_axis=num_x_axis, num_y_axis=num_y_axis, 
                                        figsize=(fig_size, fig_size))
                    plt.title(f"Biplot - Plan PC{num_x_axis}-PC{num_y_axis}", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig_biplot)
                
                with tab4:
                    st.write("**Biplot Amélioré avec mise à l'échelle des variables**")
                    
                    # Fonction pour créer un biplot personnalisé amélioré
                    def create_improved_biplot(acp, num_x_axis, num_y_axis, figsize=(10, 8)):
                        fig, ax = plt.subplots(figsize=figsize)
                        
                        # Récupérer les coordonnées des individus
                        individuals_coords = acp.row_coord_
                        x_ind = individuals_coords[:, num_x_axis-1]
                        y_ind = individuals_coords[:, num_y_axis-1]
                        
                        # Récupérer les coordonnées des variables (corrélations avec les axes)
                        variables_coords = acp.col_coord_
                        x_var = variables_coords[:, num_x_axis-1]
                        y_var = variables_coords[:, num_y_axis-1]
                        
                        # Calculer les facteurs d'échelle pour améliorer la visibilité
                        scale_factor_x = max(abs(x_ind)) / max(abs(x_var)) * 0.8
                        scale_factor_y = max(abs(y_ind)) / max(abs(y_var)) * 0.8
                        scale_factor = min(scale_factor_x, scale_factor_y)
                        
                        # Mise à l'échelle des variables
                        x_var_scaled = x_var * scale_factor
                        y_var_scaled = y_var * scale_factor
                        
                        # Tracer les individus
                        scatter = ax.scatter(x_ind, y_ind, alpha=0.6, s=50, c='blue', label='Observations')
                        
                        # Ajouter les labels des individus si pas trop nombreux
                        if len(x_ind) <= 20:
                            for i, label in enumerate(acp.row_labels_):
                                ax.annotate(str(label), (x_ind[i], y_ind[i]), 
                                        xytext=(5, 5), textcoords='offset points',
                                        fontsize=8, alpha=0.7)
                        
                        # Tracer les variables comme des flèches
                        for i, var_name in enumerate(acp.col_labels_):
                            ax.arrow(0, 0, x_var_scaled[i], y_var_scaled[i], 
                                    head_width=0.05, head_length=0.1, 
                                    fc='red', ec='red', alpha=0.8, linewidth=2)
                            
                            # Positionner les labels des variables
                            offset_x = x_var_scaled[i] * 1.1
                            offset_y = y_var_scaled[i] * 1.1
                            ax.annotate(var_name, (offset_x, offset_y), 
                                    fontsize=10, fontweight='bold', 
                                    color='red', ha='center', va='center')
                        
                        # Ajouter les axes
                        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
                        
                        # Personnaliser le graphique
                        ax.set_xlabel(f'PC{num_x_axis} ({variance_explained[num_x_axis-1]:.1f}%)', fontsize=12)
                        ax.set_ylabel(f'PC{num_y_axis} ({variance_explained[num_y_axis-1]:.1f}%)', fontsize=12)
                        ax.set_title(f'Biplot Amélioré - Plan PC{num_x_axis}-PC{num_y_axis}\n'
                                f'Variance expliquée: {variance_explained[num_x_axis-1] + variance_explained[num_y_axis-1]:.1f}%', 
                                fontsize=14, pad=20)
                        
                        # Ajouter une légende
                        from matplotlib.lines import Line2D
                        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor='blue', markersize=8, 
                                                alpha=0.6, label='Observations'),
                                        Line2D([0], [0], color='red', linewidth=2, 
                                                label='Variables')]
                        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
                        
                        # Ajuster les limites pour une meilleure visualisation
                        margin = 0.1
                        x_range = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
                        y_range = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
                        max_range = max(x_range, y_range)
                        ax.set_xlim(-max_range * (1 + margin), max_range * (1 + margin))
                        ax.set_ylim(-max_range * (1 + margin), max_range * (1 + margin))
                        
                        # Améliorer l'apparence
                        ax.grid(True, alpha=0.3)
                        ax.set_aspect('equal')
                        plt.tight_layout()
                        
                        return fig
                    
                    # Créer et afficher le biplot amélioré
                    try:
                        fig_improved = create_improved_biplot(acp, num_x_axis, num_y_axis, 
                                                            figsize=(fig_size, fig_size))
                        st.pyplot(fig_improved)
                        
                        # Informations sur l'amélioration
                        st.info("💡 Ce biplot amélioré utilise une mise à l'échelle adaptative pour une meilleure visibilité des variables")
                        
                    except Exception as e:
                        st.error(f"Erreur lors de la création du biplot amélioré: {e}")
                        st.write("Utilisation du biplot standard...")
                        fig_biplot = acp.mapping(num_x_axis=num_x_axis, num_y_axis=num_y_axis, 
                                            figsize=(fig_size, fig_size))
                        plt.title(f"Biplot - Plan PC{num_x_axis}-PC{num_y_axis}", fontsize=14)
                        plt.tight_layout()
                        st.pyplot(fig_biplot)
                
                # Contributions et qualité
                st.markdown("---")
                st.subheader("📊 Contributions et qualité de représentation")
                
                # Tabs pour chaque composante
                tabs_axes = st.tabs([f"PC{i+1}" for i in range(min(3, n_components))])
                
                for i, tab in enumerate(tabs_axes):
                    with tab:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Contributions des individus - PC{i+1}**")
                            try:
                                fig_contrib_row = acp.plot_row_contrib(num_axis=i+1, nb_values=10)
                                plt.title(f"Contributions individus - PC{i+1}", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig_contrib_row)
                            except:
                                st.warning("Impossible d'afficher les contributions des individus")
                            
                            st.write(f"**Contributions des variables - PC{i+1}**")
                            try:
                                fig_contrib_col = acp.plot_col_contrib(num_axis=i+1, nb_values=10)
                                plt.title(f"Contributions variables - PC{i+1}", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig_contrib_col)
                            except:
                                st.warning("Impossible d'afficher les contributions des variables")
                        
                        with col2:
                            st.write(f"**Qualité individus (Cos²) - PC{i+1}**")
                            try:
                                fig_cos2_row = acp.plot_row_cos2(num_axis=i+1, nb_values=10)
                                plt.title(f"Qualité individus - PC{i+1}", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig_cos2_row)
                            except:
                                st.warning("Impossible d'afficher la qualité des individus")
                            
                            st.write(f"**Qualité variables (Cos²) - PC{i+1}**")
                            try:
                                fig_cos2_col = acp.plot_col_cos2(num_axis=i+1, nb_values=10)
                                plt.title(f"Qualité variables - PC{i+1}", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig_cos2_col)
                            except:
                                st.warning("Impossible d'afficher la qualité des variables")
                
                # Tableaux de résultats détaillés
                st.markdown("---")
                st.subheader("📋 Résultats détaillés")
                
                tab1, tab2 = st.tabs(["Individus", "Variables"])
                
                with tab1:
                    try:
                        df_rows = acp.row_topandas()
                        st.dataframe(df_rows, use_container_width=True)
                        
                        # Téléchargement
                        csv_rows = df_rows.to_csv(index=True)
                        st.download_button(
                            label="📥 Télécharger résultats individus (CSV)",
                            data=csv_rows,
                            file_name="acp_individus.csv",
                            mime="text/csv",
                            key="acp_download_individuals"
                        )
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage des résultats des individus: {e}")
                
                with tab2:
                    try:
                        df_cols = acp.col_topandas()
                        st.dataframe(df_cols, use_container_width=True)
                        
                        # Téléchargement
                        csv_cols = df_cols.to_csv(index=True)
                        st.download_button(
                            label="📥 Télécharger résultats variables (CSV)",
                            data=csv_cols,
                            file_name="acp_variables.csv",
                            mime="text/csv",
                            key="acp_download_variables"
                        )
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage des résultats des variables: {e}")
                
                # Aide à l'interprétation
                st.markdown("---")
                st.subheader("🤖 Aide à l'interprétation")
                
                total_variance_2d = variance_explained[num_x_axis-1] + variance_explained[num_y_axis-1]
                
                if total_variance_2d > 70:
                    quality = "Excellente"
                    color = "🟢"
                elif total_variance_2d > 50:
                    quality = "Bonne"
                    color = "🟡"
                else:
                    quality = "Moyenne"
                    color = "🔴"
                
                st.markdown(f"**Qualité de la représentation 2D:** {color} {quality} ({total_variance_2d:.1f}% de variance)")
                
                # Variables les mieux représentées
                try:
                    df_vars = acp.col_topandas()
                    best_vars_pc1 = df_vars.nlargest(3, f'Cos2_{num_x_axis}').index.tolist()
                    best_vars_pc2 = df_vars.nlargest(3, f'Cos2_{num_y_axis}').index.tolist()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Variables les mieux représentées sur PC{num_x_axis}:**")
                        for var in best_vars_pc1:
                            st.write(f"• {var}")
                    
                    with col2:
                        st.write(f"**Variables les mieux représentées sur PC{num_y_axis}:**")
                        for var in best_vars_pc2:
                            st.write(f"• {var}")
                except:
                    pass
                
                # Conseils d'interprétation
                with st.expander("💡 Conseils d'interprétation du Biplot Amélioré"):
                    st.markdown("""
                    **Comment interpréter le Biplot Amélioré :**
                    
                    1. **Flèches rouges (Variables)** : 
                    - Plus la flèche est longue, mieux la variable est représentée
                    - L'angle entre flèches indique la corrélation (angle aigu = corrélation positive)
                    - Variables perpendiculaires = non corrélées
                    
                    2. **Points bleus (Observations)** :
                    - Proximité = profils similaires
                    - Position relative aux flèches = valeurs élevées/faibles pour ces variables
                    
                    3. **Interprétation spatiale** :
                    - Observations dans la direction d'une flèche = valeurs élevées pour cette variable
                    - Observations opposées à une flèche = valeurs faibles pour cette variable
                    
                    4. **Améliorations apportées** :
                    - Mise à l'échelle adaptative des variables pour une meilleure visibilité
                    - Flèches colorées et étiquetées clairement
                    - Axes proportionnels pour éviter les déformations
                    - Légende explicative
                    
                    **Seuils d'interprétation :**
                    - Longueur de flèche > 0.7 : Variable bien représentée
                    - Angle < 30° : Corrélation forte positive
                    - Angle > 150° : Corrélation forte négative
                    - Angle ≈ 90° : Variables indépendantes
                    """)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'exécution de l'ACP: {str(e)}")
                st.error("Vérifiez que vos données contiennent uniquement des variables numériques et que la bibliothèque fanalysis est installée.")
        
        else:
            st.info("👆 Configurez vos paramètres et cliquez sur 'Lancer l'Analyse ACP' pour commencer")

    def ACM_analysis(df):
        """
        Fonction d'Analyse des Correspondances Multiples intégrée
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        # Vérifier si la bibliothèque fanalysis est disponible
        try:
            from fanalysis.mca import MCA
        except ImportError:
            st.error("❌ La bibliothèque 'fanalysis' n'est pas installée. Installez-la avec: pip install fanalysis")
            st.info("💡 Alternative: Vous pouvez utiliser sklearn.decomposition pour une version simplifiée de l'ACM")
            return
        
        from scipy.stats import chi2_contingency, chi2
        from sklearn.preprocessing import LabelEncoder
        
        # Fonction pour calculer le V de Cramér
        def cramers_v(x, y):
            """Calcule le V de Cramér entre deux variables catégorielles"""
            contingency_table = pd.crosstab(x, y)
            chi2_stat, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape) - 1
            if min_dim == 0:
                return 0
            return np.sqrt(chi2_stat / (n * min_dim))
        
        st.header("🔍 Analyse des Correspondances Multiples (ACM)")
        st.markdown("Analysez les relations entre variables catégorielles")
        
        if df.empty:
            st.warning("⚠️ Aucune donnée disponible")
            return
        
        # Configuration en colonnes
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ Configuration")
            
            # Gestion des valeurs manquantes
            missing_method = st.selectbox(
                "Gestion des valeurs manquantes:",
                ["Supprimer les lignes", "Remplacer par mode", "Remplacer par 'Missing'", "Ne rien faire"],
                help="Choisissez comment traiter les valeurs manquantes"
            )
            
            # Fréquence minimale
            min_freq = st.slider(
                "Fréquence minimale des modalités (%)",
                min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                help="Modalités avec une fréquence inférieure seront regroupées"
            )
            
            # Index personnalisé
            use_custom_index = st.checkbox("Utiliser une colonne comme index")
            if use_custom_index:
                index_col = st.selectbox(
                    "Colonne pour l'index:",
                    options=df.columns.tolist(),
                    help="Cette colonne sera utilisée comme identifiant"
                )
        
        with col2:
            st.subheader("📊 Aperçu des données")
            st.write("**Premières lignes:**")
            st.dataframe(df.head())
            
            # Informations sur les données
            info_data = {
                'Type': df.dtypes.astype(str),
                'Valeurs manquantes': df.isnull().sum(),
                '% manquantes': (df.isnull().sum() / len(df) * 100).round(2)
            }
            info_df = pd.DataFrame(info_data)
            with st.expander("Informations détaillées"):
                st.dataframe(info_df)
        
        # Préprocessing des données
        df_processed = df.copy()
        
        # Appliquer l'index personnalisé
        if use_custom_index and 'index_col' in locals():
            df_processed.set_index(index_col, inplace=True)
            st.info(f"Index défini sur la colonne: {index_col}")
        
        # Traitement des valeurs manquantes
        if missing_method == "Supprimer les lignes":
            df_processed = df_processed.dropna()
            st.info(f"Lignes supprimées: {len(df) - len(df_processed)}")
        elif missing_method == "Remplacer par mode":
            for col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    mode_val = df_processed[col].mode()
                    if len(mode_val) > 0:
                        df_processed[col].fillna(mode_val[0], inplace=True)
        elif missing_method == "Remplacer par 'Missing'":
            df_processed = df_processed.fillna('Missing')
        
        # Sélection des variables
        st.markdown("---")
        st.subheader("📋 Sélection des variables")
        
        # Détecter automatiquement les variables catégorielles
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Interface pour sélection des variables
        col1, col2 = st.columns(2)
        
        with col1:
            vars_selected = st.multiselect(
                "Variables pour l'analyse:",
                options=df_processed.columns.tolist(),
                default=categorical_cols[:5] if len(categorical_cols) >= 5 else categorical_cols,
                help="Sélectionnez les variables catégorielles"
            )
        
        with col2:
            if vars_selected:
                st.write("**Variables sélectionnées:**")
                for var in vars_selected:
                    unique_count = df_processed[var].nunique()
                    st.write(f"• {var}: {unique_count} modalités")
        
        if not vars_selected:
            st.warning("⚠️ Veuillez sélectionner au moins une variable")
            return
        
        # Vérification des types de données
        non_categorical = []
        for var in vars_selected:
            if df_processed[var].dtype not in ['object', 'category']:
                non_categorical.append(var)
        
        if non_categorical:
            st.warning(f"⚠️ Variables non-catégorielles détectées: {non_categorical}")
            convert_numeric = st.checkbox("Convertir les variables numériques en catégorielles")
            if convert_numeric:
                for var in non_categorical:
                    n_bins = st.slider(f"Nombre de classes pour {var}", 3, 10, 5, key=f"bins_{var}")
                    df_processed[var] = pd.cut(df_processed[var], bins=n_bins, duplicates='drop')
                    df_processed[var] = df_processed[var].astype(str)
        
        # Créer le dataset final
        X = df_processed[vars_selected].copy()
        
        # Traitement des modalités rares
        if min_freq > 0:
            for col in X.columns:
                freq = X[col].value_counts(normalize=True) * 100
                rare_categories = freq[freq < min_freq].index
                if len(rare_categories) > 0:
                    X[col] = X[col].replace(rare_categories, 'Autres')
                    st.info(f"Modalités regroupées dans '{col}': {len(rare_categories)}")
        
        # Encodage
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].astype(str)

        # Paramètres
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Nombre de composantes", 2, min(10, len(vars_selected)), 5)
        with col2:
            fig_size = st.slider("Taille des graphiques", 6, 12, 8)
        
        st.markdown("---")
        
        # Bouton pour lancer l'analyse
        if st.button("🚀 Lancer l'Analyse ACM", type="primary"):
            
            # Affichage des données finales
            st.subheader("✅ Données préparées")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Aperçu final:**")
                st.dataframe(X.head())
            
            with col2:
                st.write("**Résumé des modalités:**")
                summary_data = []
                for col in X.columns:
                    unique_vals = X[col].nunique()
                    most_frequent = X[col].value_counts().iloc[0]
                    summary_data.append({
                        'Variable': col,
                        'Modalités': unique_vals,
                        'Plus fréquente': f"{most_frequent} obs."
                    })
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df)
            
            # Tests statistiques préliminaires
            if len(vars_selected) >= 2:
                st.markdown("---")
                st.subheader("📈 Tests statistiques préliminaires")
                
                chi2_results = []
                cramers_results = []
                
                for i, var1 in enumerate(vars_selected):
                    for j, var2 in enumerate(vars_selected):
                        if i < j:
                            contingency_table = pd.crosstab(X[var1], X[var2])
                            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                            cramers_v_value = cramers_v(X[var1], X[var2])
                            
                            chi2_results.append({
                                'Variable 1': var1,
                                'Variable 2': var2,
                                'Chi-2': round(chi2_stat, 4),
                                'p-value': round(p_value, 6),
                                'Significatif': p_value < 0.05
                            })
                            
                            cramers_results.append({
                                'Variable 1': var1,
                                'Variable 2': var2,
                                'V de Cramér': round(cramers_v_value, 4),
                                'Force': 'Forte' if cramers_v_value > 0.3 else 'Modérée' if cramers_v_value > 0.1 else 'Faible'
                            })
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Tests du Chi-2:**")
                    chi2_df = pd.DataFrame(chi2_results)
                    st.dataframe(chi2_df, use_container_width=True)
                
                with col2:
                    st.write("**V de Cramér:**")
                    cramers_df = pd.DataFrame(cramers_results)
                    st.dataframe(cramers_df, use_container_width=True)
            
            # Test d'adéquation
            st.markdown("---")
            st.subheader("🧪 Adéquation des données")
            
            # Matrice indicatrice et corrélations
            X_encoded = pd.get_dummies(X)
            correlation_matrix = X_encoded.corr()
            mean_correlation = correlation_matrix.abs().mean().mean()
            
            if mean_correlation > 0.5:
                adequacy = "Excellente"
                color = "🟢"
            elif mean_correlation > 0.3:
                adequacy = "Bonne"
                color = "🟡"
            else:
                adequacy = "Faible"
                color = "🔴"
            
            st.markdown(f"**Adéquation:** {color} {adequacy} (Corrélation moyenne: {mean_correlation:.3f})")
            
            # Exécution de l'ACM
            st.markdown("---")
            st.subheader("🎯 Résultats de l'ACM")
            
            try:
                
                
                # Exécution ACM
                acm = MCA(row_labels=X.index.values, var_labels=X.columns)
                acm.fit(X.values)
                
                # Valeurs propres
                st.subheader("📊 Valeurs propres et variance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_eigenvalues = acm.plot_eigenvalues()
                    plt.title("Valeurs propres", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig_eigenvalues, use_container_width=True)
                
                with col2:
                    fig_cumulative = acm.plot_eigenvalues("cumulative")
                    plt.title("Variance cumulée", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig_cumulative, use_container_width=True)
                
                # Tableau des valeurs propres
                eigenvalues = acm.eig_[1]
                variance_explained = (eigenvalues / eigenvalues.sum()) * 100
                cumulative_variance = np.cumsum(variance_explained)
                
                eigenvalues_df = pd.DataFrame({
                    'Composante': [f'Dim {i+1}' for i in range(len(eigenvalues))],
                    'Valeur propre': eigenvalues.round(4),
                    'Variance (%)': variance_explained.round(2),
                    'Variance cumulée (%)': cumulative_variance.round(2)
                })
                
                st.write("**Tableau des valeurs propres:**")
                st.dataframe(eigenvalues_df, use_container_width=True)
                
                # Critère de Kaiser
                kaiser_threshold = 1 / len(vars_selected)
                significant_kaiser = eigenvalues > kaiser_threshold
                st.info(f"Critère de Kaiser (seuil: {kaiser_threshold:.3f}): {sum(significant_kaiser)} axes significatifs")
                
                # Sélection des axes pour visualisation
                st.markdown("---")
                st.subheader("📈 Visualisations")
                
                col1, col2 = st.columns(2)
                with col1:
                    axis_x = st.selectbox("Axe X", [f"Axe {i+1}" for i in range(n_components)], index=0)
                    num_x_axis = int(axis_x.split()[1])
                with col2:
                    axis_y = st.selectbox("Axe Y", [f"Axe {i+1}" for i in range(n_components)], index=1)
                    num_y_axis = int(axis_y.split()[1])
                
                # Graphiques principaux
                tab1, tab2, tab3 = st.tabs(["👥 Individus", "🏷️ Modalités", "🎭 Biplot"])
                
                with tab1:
                    st.write("**Projection des individus**")
                    fig_rows = acm.mapping_row(num_x_axis=num_x_axis, num_y_axis=num_y_axis, 
                                            figsize=(fig_size, fig_size))
                    plt.title(f"Individus - Plan {num_x_axis}-{num_y_axis}", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig_rows, use_container_width=True)
                
                with tab2:
                    st.write("**Projection des modalités**")
                    fig_cols = acm.mapping_col(num_x_axis=num_x_axis, num_y_axis=num_y_axis,
                                            figsize=(fig_size, fig_size))
                    plt.title(f"Modalités - Plan {num_x_axis}-{num_y_axis}", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig_cols, use_container_width=True)
                
                with tab3:
                    st.write("**Biplot (Individus + Modalités)**")
                    fig_biplot = acm.mapping(num_x_axis=num_x_axis, num_y_axis=num_y_axis, 
                                        short_labels=False, figsize=(fig_size, fig_size))
                    plt.title(f"Biplot - Plan {num_x_axis}-{num_y_axis}", fontsize=14)
                    plt.tight_layout()
                    st.pyplot(fig_biplot, use_container_width=True)
                
                # Contributions et qualité
                st.markdown("---")
                st.subheader("📊 Contributions et qualité de représentation")
                
                # Tabs pour chaque axe
                tabs_axes = st.tabs([f"Axe {i+1}" for i in range(min(3, n_components))])
                
                for i, tab in enumerate(tabs_axes):
                    with tab:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Contributions des individus - Axe {i+1}**")
                            try:
                                fig_contrib_row = acm.plot_row_contrib(num_axis=i+1, nb_values=10)
                                plt.title(f"Contributions individus - Axe {i+1}", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig_contrib_row, use_container_width=True)
                            except:
                                st.warning("Impossible d'afficher les contributions des individus")
                            
                            st.write(f"**Contributions des modalités - Axe {i+1}**")
                            try:
                                fig_contrib_col = acm.plot_col_contrib(num_axis=i+1, nb_values=10)
                                plt.title(f"Contributions modalités - Axe {i+1}", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig_contrib_col, use_container_width=True)
                            except:
                                st.warning("Impossible d'afficher les contributions des modalités")
                        
                        with col2:
                            st.write(f"**Qualité individus (Cos²) - Axe {i+1}**")
                            try:
                                fig_cos2_row = acm.plot_row_cos2(num_axis=i+1, nb_values=10)
                                plt.title(f"Qualité individus - Axe {i+1}", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig_cos2_row, use_container_width=True)
                            except:
                                st.warning("Impossible d'afficher la qualité des individus")
                            
                            st.write(f"**Qualité modalités (Cos²) - Axe {i+1}**")
                            try:
                                fig_cos2_col = acm.plot_col_cos2(num_axis=i+1, nb_values=10)
                                plt.title(f"Qualité modalités - Axe {i+1}", fontsize=12)
                                plt.tight_layout()
                                st.pyplot(fig_cos2_col, use_container_width=True)
                            except:
                                st.warning("Impossible d'afficher la qualité des modalités")
                
                # Tableaux de résultats détaillés
                st.markdown("---")
                st.subheader("📋 Résultats détaillés")
                
                tab1, tab2 = st.tabs(["Individus", "Modalités"])
                
                with tab1:
                    try:
                        df_rows = acm.row_topandas()
                        st.dataframe(df_rows, use_container_width=True)
                        
                        # Téléchargement
                        csv_rows = df_rows.to_csv(index=True)
                        st.download_button(
                            label="📥 Télécharger résultats individus (CSV)",
                            data=csv_rows,
                            file_name="acm_individus.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage des résultats des individus: {e}")
                
                with tab2:
                    try:
                        df_cols = acm.col_topandas()
                        st.dataframe(df_cols, use_container_width=True)
                        
                        # Téléchargement
                        csv_cols = df_cols.to_csv(index=True)
                        st.download_button(
                            label="📥 Télécharger résultats modalités (CSV)",
                            data=csv_cols,
                            file_name="acm_modalites.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Erreur lors de l'affichage des résultats des modalités: {e}")
                
                # Aide à l'interprétation
                st.markdown("---")
                st.subheader("🤖 Aide à l'interprétation")
                
                total_variance_2d = variance_explained[num_x_axis-1] + variance_explained[num_y_axis-1]
                
                if total_variance_2d > 50:
                    quality = "Excellente"
                    color = "🟢"
                elif total_variance_2d > 30:
                    quality = "Bonne"
                    color = "🟡"
                else:
                    quality = "Moyenne"
                    color = "🔴"
                
                st.markdown(f"**Qualité de la représentation 2D:** {color} {quality} ({total_variance_2d:.1f}% de variance)")
                
                # Conseils d'interprétation
                with st.expander("💡 Conseils d'interprétation"):
                    st.markdown("""
                    **Comment interpréter les résultats de l'ACM :**
                    
                    1. **Valeurs propres** : Plus elles sont élevées, plus l'axe explique de variance
                    2. **Contributions** : Indiquent quels individus/modalités contribuent le plus à la formation des axes
                    3. **Qualité (Cos²)** : Mesure la qualité de représentation (0-1, plus c'est proche de 1, mieux c'est)
                    4. **Proximité sur le graphique** : Individus/modalités proches ont des profils similaires
                    5. **Distance à l'origine** : Plus un point est éloigné, plus il est atypique
                    
                    **Seuils indicatifs :**
                    - Contribution moyenne : 100/nombre d'éléments %
                    - Qualité acceptable : > 0.3
                    - Association forte (V de Cramér) : > 0.3
                    """)
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'exécution de l'ACM: {str(e)}")
                st.error("Vérifiez que vos données contiennent uniquement des variables catégorielles et que la bibliothèque fanalysis est installée.")
        
        else:
            st.info("👆 Configurez vos paramètres et cliquez sur 'Lancer l'Analyse ACM' pour commencer")


    def AFC_analysis(df):
        """
        Analyse Factorielle des Correspondances avec interface utilisateur améliorée
        """
        # Vérification de la disponibilité de fanalysis
        try:
            from fanalysis.ca import CA
            fanalysis_available = True
        except ImportError:
            fanalysis_available = False
            st.error("📦 Le package 'fanalysis' n'est pas installé. Installez-le avec: `pip install fanalysis`")
            return
        
        # Interface utilisateur moderne avec sidebar
        st.title("🔍 Analyse Factorielle des Correspondances (AFC)")
        
        # Configuration dans la sidebar
        with st.sidebar:
            st.header("⚙️ Configuration de l'analyse")
            
            # Aperçu des données
            with st.expander("📊 Aperçu des données", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.caption(f"Dataset: {df.shape[0]} lignes × {df.shape[1]} colonnes")
        
        if df.empty:
            st.warning("⚠️ Aucune donnée n'est disponible pour l'analyse.")
            return
        
        # Analyse des types de variables
        categorical_cols = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Interface de sélection des variables améliorée
        st.subheader("📋 Sélection des variables")
        
        # Options de conversion
        col1, col2 = st.columns([2, 1])
        with col1:
            convert_numeric = st.checkbox(
                "🔄 Convertir les variables numériques en catégorielles", 
                value=False,
                help="Permet d'analyser des variables numériques en les discrétisant"
            )
        
        # Déterminer les colonnes disponibles
        if convert_numeric:
            available_cols = df.columns.tolist()
            st.info(f"📊 {len(categorical_cols)} variables catégorielles + {len(numerical_cols)} variables numériques disponibles")
        else:
            available_cols = categorical_cols
            if not available_cols:
                st.error("❌ Aucune variable catégorielle trouvée. Activez la conversion des variables numériques.")
                return
            st.info(f"📊 {len(categorical_cols)} variables catégorielles disponibles")
        
        # Sélection des variables avec validation
        col1, col2 = st.columns(2)
        with col1:
            var1 = st.selectbox(
                "🎯 Première variable", 
                available_cols, 
                index=0 if available_cols else None,
                help="Sélectionnez la première variable pour l'analyse"
            )
        
        with col2:
            var2_options = [col for col in available_cols if col != var1]
            var2 = st.selectbox(
                "🎯 Deuxième variable", 
                var2_options, 
                index=0 if var2_options else None,
                help="Sélectionnez la deuxième variable pour l'analyse"
            )
        
        # Paramètres de discrétisation (si nécessaire)
        if convert_numeric and (var1 in numerical_cols or var2 in numerical_cols):
            with st.expander("⚙️ Paramètres de discrétisation", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    n_bins = st.slider(
                        "Nombre de catégories", 
                        min_value=3, max_value=10, value=5,
                        help="Nombre de catégories pour la discrétisation"
                    )
                with col2:
                    discretization_method = st.selectbox(
                        "Méthode de discrétisation", 
                        ["cut", "qcut"],
                        format_func=lambda x: "Intervalles égaux" if x == "cut" else "Quantiles égaux",
                        help="cut: intervalles de taille égale\nqcut: intervalles avec effectifs égaux"
                    )
        
        # Validation des sélections
        if not var1 or not var2 or var1 == var2:
            st.warning("⚠️ Veuillez sélectionner deux variables différentes pour continuer.")
            return
        
        # Configuration des graphiques dans un expander
        with st.expander("🎨 Personnalisation des graphiques", expanded=False):
            st.subheader("Configuration des titres et axes")
            
            # Organisation en onglets pour la configuration
            config_tab1, config_tab2, config_tab3,config_tab4,config_tab5 = st.tabs(["Valeurs propres", "Modalités", "Association","contributions","Décomposition du Chi-2"])
            
            with config_tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Graphique des valeurs propres**")
                    title_eigenvalues = st.text_input("Titre", "Valeurs propres des composantes", key="title_eigen")
                    xaxis_title_eigenvalues = st.text_input("Axe X", "Composantes", key="x_eigen")
                    yaxis_title_eigenvalues = st.text_input("Axe Y", "Valeurs propres", key="y_eigen")
                    title_x_eigenvalues = st.slider("Position du titre X", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")

                with col2:
                    st.write("**Graphique cumulé**")
                    title_cumulative = st.text_input("Titre", "Valeurs propres cumulées", key="title_cum")
                    xaxis_title_cumulative = st.text_input("Axe X", "Composantes", key="x_cum")
                    yaxis_title_cumulative = st.text_input("Axe Y", "Valeurs cumulées", key="y_cum")
                    title_x_cumulative = st.slider("Position du titre X (cumulées)", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")
            
            with config_tab2:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Modalités lignes**")
                    title_row = st.text_input("Titre", "Modalités lignes", key="title_row")
                    xaxis_title_row = st.text_input("Axe X", "Dim 1", key="x_row")
                    yaxis_title_row = st.text_input("Axe Y", "Dim 2", key="y_row")
                    title_x_row = st.slider("Position du titre X (lignes)", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")

                with col2:
                    st.write("**Modalités colonnes**")
                    title_col = st.text_input("Titre", "Modalités colonnes", key="title_col")
                    xaxis_title_col = st.text_input("Axe X", "Dim 1", key="x_col")
                    yaxis_title_col = st.text_input("Axe Y", "Dim 2", key="y_col")
                    title_x_col = st.slider("Position du titre X (colonnes)", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")
            
            with config_tab3:
                st.write("**Association lignes-colonnes**")
                title_association = st.text_input("Titre", "Association lignes-colonnes", key="title_assoc")
                xaxis_title_association = st.text_input("Axe X", "Dim 1", key="x_assoc")
                yaxis_title_association = st.text_input("Axe Y", "Dim 2", key="y_assoc")
                title_x_association = st.slider("Position du titre X (association)", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")
            with config_tab4:
                st.write("**Contributions**")
                st.write("Configurez les titres et axes pour les graphiques de contributions lignes et colonnes")
                st.write("** Contributions des lignes **")
                coll1,coll2 = st.columns(2)
                with coll1:
                    st.write("**Axe 1**")
                    title_contrib1_row = st.text_input("Titre", "Contributions des lignes à la dimension 1", key="title_contrib1_row")
                    xaxis_title_contrib1_row = st.text_input("Axe X", "Modalités", key="x_contrib1_row")
                    yaxis_title_contrib1row = st.text_input("Axe Y", "Contributions (%)", key="y_contrib1_row")
                    title_x_contrib1_row = st.slider("Position du titre X (Axe 1 ligne)", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")
                with coll2:
                    st.write("**Axe 2**")
                    title_contrib2_row = st.text_input("Titre", "Contributions des lignes à la dimension 2", key="title_contrib2_row")
                    xaxis_title_contrib2_row = st.text_input("Axe X", "Modalités", key="x_contrib2_row")
                    yaxis_title_contrib2_row = st.text_input("Axe Y", "Contributions", key="y_contrib2_row")
                    title_x_contrib2_row = st.slider("Position du titre X (Axe 2 ligne)", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")
                st.write("** Contributions des colonnes **")
                coll3,coll4 = st.columns(2)
                with coll3:
                    st.write("**Axe 1**")
                    title_contrib1_col = st.text_input("Titre", "Contributions des colonnes à la dimension 1", key="title_contrib1_col")
                    xaxis_title_contrib1_col = st.text_input("Axe X", "Modalités", key="x_contrib1_col")
                    yaxis_title_contrib1_col = st.text_input("Axe Y", "Contributions", key="y_contrib1_col")
                    title_x_contrib1_col = st.slider("Position du titre X (Axe 1 colonne)", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")
                with coll4:
                    st.write("**Axe 2**")
                    title_contrib2_col = st.text_input("Titre", "Contributions des colonnes à la dimension 2", key="title_contrib2_col")
                    xaxis_title_contrib2_col = st.text_input("Axe X", "Modalités", key="x_contrib2_col")
                    yaxis_title_contrib2_col = st.text_input("Axe Y", "Contributions", key="y_contrib2_col")
                    title_x_contrib2_col = st.slider("Position du titre X (Axe 2 colonne)", 0.0, 1.0, 0.5, step=0.01, help="Position du titre sur l'axe X (0.0 = gauche, 1.0 = droite)")
                
            with config_tab5:
                col5, col6 = st.columns(2)
                with col5:
                    st.write("**Décomposition du Chi-2**")
                    title_chi2 = st.text_input("Titre", "Décomposition du Chi-2 (fraction)", key="title_chi2")
                    cmap_chi2 = st.selectbox(
                        "Palette de couleurs",
                        options=['YlOrRd','RdBu_r',"viridis", "plasma", "inferno", "magma", "cividis"],

                        index=0,
                        help="Sélectionnez la palette de couleurs pour le graphique de décomposition du Chi-2"
                    )
                with col6:
                    st.write("**Résidus standardisés**")
                    title_residuals = st.text_input("Titre", "Résidus standardisés", key="title_residuals")
                    cmap_residuals = st.selectbox(
                        "Palette de couleurs",
                        options=['RdBu_r','YlOrRd',"viridis", "plasma", "inferno", "magma", "cividis"],
                        index=0,
                        help="Sélectionnez la palette de couleurs pour le graphique des résidus standardisés"
                    )


        # Bouton d'exécution principal
        st.divider()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_analysis = st.button(
                "🚀 Lancer l'analyse AFC", 
                type="primary", 
                use_container_width=True,
                help="Cliquez pour démarrer l'analyse factorielle des correspondances"
            )
        
        if not run_analysis:
            return
        
        # Exécution de l'analyse
        with st.spinner("🔄 Analyse en cours... Veuillez patienter."):
            try:
                # Préparation des données
                df_work = df.copy()
                
                # Conversion des variables numériques si nécessaire
                conversion_messages = []
                if convert_numeric:
                    for var in [var1, var2]:
                        if var in numerical_cols:
                            try:
                                if discretization_method == "cut":
                                    df_work[var] = pd.cut(df_work[var], bins=n_bins, precision=2)
                                else:
                                    df_work[var] = pd.qcut(df_work[var], q=n_bins, precision=2, duplicates='drop')
                                
                                df_work[var] = df_work[var].astype(str)
                                conversion_messages.append(f"✅ '{var}' convertie en {n_bins} catégories")
                            except Exception as e:
                                st.error(f"❌ Erreur lors de la conversion de '{var}': {e}")
                                return
                
                # Affichage des messages de conversion
                if conversion_messages:
                    for msg in conversion_messages:
                        st.success(msg)
                
                # Création du tableau de contingence
                df_cont = pd.crosstab(df_work[var1], df_work[var2])
                
                if df_cont.size <= 1:
                    st.error("❌ Le tableau de contingence est invalide. Vérifiez vos variables.")
                    return
                
                # Test du Chi-2
                from scipy.stats import chi2_contingency
                res = chi2_contingency(df_cont.values, correction=False)
                
                # Affichage des résultats principaux
                st.success("✅ Analyse terminée avec succès!")
                
                # Métriques principales en haut
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Chi-2", f"{res.statistic:.2f}")
                with metric_col2:
                    st.metric("p-value", f"{res.pvalue:.4f}")
                with metric_col3:
                    significance = "Significatif" if res.pvalue < 0.05 else "Non significatif"
                    st.metric("Test", significance)
                with metric_col4:
                    st.metric("Degrés liberté", f"{res.dof}")
                
                # Onglets pour les résultats
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Tableaux", "📈 Valeurs propres", "🎯 Modalités", 
                    "📋 Contributions", "🔥 Résidus & Chi-2"
                ])
                
                with tab1:
                    st.subheader("📋 Tableau de contingence")
                    st.dataframe(df_cont, use_container_width=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📊 Profils lignes")
                        profil_ligne = df_cont.divide(df_cont.sum(axis=1), axis=0)
                        st.dataframe(profil_ligne.round(3), use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Profils colonnes")
                        profil_colonne = df_cont.divide(df_cont.sum(axis=0), axis=1)
                        st.dataframe(profil_colonne.round(3), use_container_width=True)
                
                # AFC avec fanalysis
                if fanalysis_available:
                    afc = CA(row_labels=df_cont.index, col_labels=df_cont.columns, stats=True)
                    afc.fit(df_cont.values)
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📈 Valeurs propres")
                            val = afc.eig_[0]
                            x = np.arange(1, len(val) + 1)
                            
                            fig_eigenvalues = px.bar(
                                x=x, y=val,
                                labels={'x': xaxis_title_eigenvalues, 'y': yaxis_title_eigenvalues},
                                title=title_eigenvalues,
                                color=val,
                                color_continuous_scale='viridis'
                            )
                            fig_eigenvalues.update_layout(
                                title_x=title_x_eigenvalues,
                                showlegend=False,
                                height=400
                            )
                            fig_eigenvalues.update_traces(texttemplate='%{y:.3f}', textposition='outside')
                            st.plotly_chart(fig_eigenvalues, use_container_width=True)
                        
                        with col2:
                            st.subheader("📊 Valeurs propres cumulées")
                            val_cum = afc.eig_[2]
                            
                            fig_cumulative = px.bar(
                                x=x, y=val_cum,
                                labels={'x': xaxis_title_cumulative, 'y': yaxis_title_cumulative},
                                title=title_cumulative,
                                color=val_cum,
                                color_continuous_scale='plasma'
                            )
                            fig_cumulative.update_layout(
                                title_x=title_x_cumulative,
                                showlegend=False,
                                height=400
                            )
                            fig_cumulative.update_traces(texttemplate='%{y:.1%}', textposition='outside')
                            st.plotly_chart(fig_cumulative, use_container_width=True)
                    
                    with tab3:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("🔴 Modalités lignes")
                            info_ligne = afc.row_topandas()
                            
                            row_graph = px.scatter(
                                info_ligne, x="row_coord_dim1", y="row_coord_dim2",
                                text=info_ligne.index,
                                labels={"row_coord_dim1": xaxis_title_row, "row_coord_dim2": yaxis_title_row},
                                title=title_row,
                                color_discrete_sequence=['red']
                            )
                            
                            # Configuration du graphique
                            all_values = np.concatenate([
                                info_ligne["row_coord_dim1"].values,
                                info_ligne["row_coord_dim2"].values
                            ])
                            max_val = np.max(np.abs(all_values))
                            axis_limit = max_val * 1.2
                            
                            row_graph.update_traces(textposition='top center', marker_size=10)
                            row_graph.update_layout(
                                title_x=title_x_row,
                                xaxis=dict(range=[-axis_limit, axis_limit], zeroline=True, zerolinecolor="black"),
                                yaxis=dict(range=[-axis_limit, axis_limit], zeroline=True, zerolinecolor="black", scaleanchor="x"),
                                plot_bgcolor="white",
                                height=500,
                                showlegend=False
                            )
                            row_graph.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
                            row_graph.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)
                            
                            st.plotly_chart(row_graph, use_container_width=True)
                        
                        with col2:
                            st.subheader("🔵 Modalités colonnes")
                            info_col = afc.col_topandas()
                            
                            col_graph = px.scatter(
                                info_col, x="col_coord_dim1", y="col_coord_dim2",
                                text=info_col.index,
                                labels={"col_coord_dim1": xaxis_title_col, "col_coord_dim2": yaxis_title_col},
                                title=title_col,
                                color_discrete_sequence=['blue']
                            )
                            
                            # Configuration similaire
                            all_values_col = np.concatenate([
                                info_col["col_coord_dim1"].values,
                                info_col["col_coord_dim2"].values
                            ])
                            max_val_col = np.max(np.abs(all_values_col))
                            axis_limit_col = max_val_col * 1.2
                            
                            col_graph.update_traces(textposition='top center', marker_size=10)
                            col_graph.update_layout(
                                title_x=title_x_col,
                                xaxis=dict(range=[-axis_limit_col, axis_limit_col], zeroline=True, zerolinecolor="black"),
                                yaxis=dict(range=[-axis_limit_col, axis_limit_col], zeroline=True, zerolinecolor="black", scaleanchor="x"),
                                plot_bgcolor="white",
                                height=500,
                                showlegend=False
                            )
                            col_graph.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
                            col_graph.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)
                            
                            st.plotly_chart(col_graph, use_container_width=True)
                        
                        # Graphique d'association combiné
                        st.subheader("🎯 Association lignes-colonnes")
                        
                        map_info = pd.concat([info_ligne, info_col], axis=0)
                        row_col = ['Modalité ligne'] * len(info_ligne) + ['Modalité colonne'] * len(info_col)
                        
                        coord_dim1 = list(info_ligne["row_coord_dim1"]) + list(info_col["col_coord_dim1"])
                        coord_dim2 = list(info_ligne["row_coord_dim2"]) + list(info_col["col_coord_dim2"])
                        
                        map_df = pd.DataFrame({
                            'Type': row_col,
                            'coord_dim1': coord_dim1,
                            'coord_dim2': coord_dim2,
                            'Label': list(info_ligne.index) + list(info_col.index)
                        })
                        
                        map_afc_graph = px.scatter(
                            map_df, x="coord_dim1", y="coord_dim2",
                            color="Type", text="Label",
                            labels={"coord_dim1": xaxis_title_association, "coord_dim2": yaxis_title_association},
                            title=title_association,
                            color_discrete_map={"Modalité ligne": "red", "Modalité colonne": "blue"}
                        )
                        
                        # Configuration du graphique combiné
                        all_combined = np.concatenate([coord_dim1, coord_dim2])
                        max_combined = np.max(np.abs(all_combined))
                        axis_limit_combined = max_combined * 1.2
                        
                        map_afc_graph.update_traces(textposition='top center', marker_size=12)
                        map_afc_graph.update_layout(
                            title_x=title_x_association,
                            xaxis=dict(range=[-axis_limit_combined, axis_limit_combined], zeroline=True, zerolinecolor="black"),
                            yaxis=dict(range=[-axis_limit_combined, axis_limit_combined], zeroline=True, zerolinecolor="black", scaleanchor="x"),
                            plot_bgcolor="white",
                            height=600,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        map_afc_graph.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1)
                        map_afc_graph.add_vline(x=0, line_dash="solid", line_color="gray", line_width=1)
                        
                        st.plotly_chart(map_afc_graph, use_container_width=True)
                    
                    with tab4:
                        st.subheader("📋 Contributions des modalités aux axes")
                        
                        contrib_col1, contrib_col2 = st.columns(2)
                

                        with contrib_col1:
                            st.write("**Axe 1**")
                            st.write("Contributions lignes - Axe 1")
                            
                            # Graphique 1 : Contributions à la dimension 1
                            fig_contrib_row_1 = px.bar(
                                info_ligne,
                                y=info_ligne.index,
                                x="row_contrib_dim1",
                                labels={"row_contrib_dim1": yaxis_title_contrib1row, "y": xaxis_title_contrib1_row},
                                title=title_contrib1_row,
                                orientation="h",
                                text="row_contrib_dim1"
                            )

                            # Personnaliser le graphique des contributions
                            fig_contrib_row_1.update_traces(
                                texttemplate='%{text:.1f}%',
                                textposition="outside",
                                textfont_size=12
                            )

                            fig_contrib_row_1.update_layout(
                                title_x=title_x_contrib1_row,
                                xaxis_title=yaxis_title_contrib1row,
                                yaxis_title=xaxis_title_contrib1_row,
                                width=800,
                                height=500
                            )

                            st.plotly_chart(fig_contrib_row_1, use_container_width=True)
                            
                            st.write("Contributions colonnes - Axe 1")

                            # Graphique 1 : Contributions à la dimension 1
                            fig_contrib_col_1 = px.bar(
                                info_col,
                                y=info_col.index,
                                x="col_contrib_dim1",
                                labels={"col_contrib_dim1": "Contribution (%)", "y": "Modalités"},
                                title=title_contrib1_col,
                                orientation="h",
                                text="col_contrib_dim1"
                            )

                            # Personnaliser le graphique des contributions
                            fig_contrib_col_1.update_traces(
                                texttemplate='%{text:.1f}%',
                                textposition="outside",
                                textfont_size=12
                            )

                            fig_contrib_col_1.update_layout(
                                title_x=title_x_contrib1_col,
                                xaxis_title=yaxis_title_contrib1_col,
                                yaxis_title=xaxis_title_contrib1_col,
                                width=800,
                                height=500
                            )

                            st.plotly_chart(fig_contrib_col_1, use_container_width=True)
                        
                        with contrib_col2:
                            st.write("**Axe 2**")
                            st.write("Contributions lignes - Axe 2")
                            # Graphique 1 : Contributions à la dimension 1
                            fig_contrib_row_2 = px.bar(
                                info_ligne,
                                y=info_ligne.index,
                                x="row_contrib_dim2",
                                labels={"row_contrib_dim1": "Contribution (%)", "y": "Modalités"},
                                title=title_contrib2_row,
                                orientation="h",
                                text="row_contrib_dim2"
                            )

                            # Personnaliser le graphique des contributions
                            fig_contrib_row_2.update_traces(
                                texttemplate='%{text:.1f}%',
                                textposition="outside",
                                textfont_size=12
                            )

                            fig_contrib_row_2.update_layout(
                                title_x=title_x_contrib2_row,
                                xaxis_title=yaxis_title_contrib2_row,
                                yaxis_title=xaxis_title_contrib2_row,
                                width=800,
                                height=500
                            )

                            st.plotly_chart(fig_contrib_row_2, use_container_width=True)
                            
                            st.write("Contributions colonnes - Axe 2")
                            fig_contrib_col_2 = px.bar(
                                info_col,
                                y=info_col.index,
                                x="col_contrib_dim2",
                                labels={"col_contrib_dim2": "Contribution (%)", "y": "Modalités"},
                                title=title_contrib2_col,
                                orientation="h",
                                text="col_contrib_dim2"
                            )

                            # Personnaliser le graphique des contributions
                            fig_contrib_col_2.update_traces(
                                texttemplate='%{text:.1f}%',
                                textposition="outside",
                                textfont_size=12
                            )

                            fig_contrib_col_2.update_layout(
                                title_x=title_x_contrib2_col,
                                xaxis_title=yaxis_title_contrib2_col,
                                yaxis_title=xaxis_title_contrib2_col,
                                width=800,
                                height=500
                            )

                            st.plotly_chart(fig_contrib_col_2, use_container_width=True)
                    
                    with tab5:
                        st.subheader("🔥 Décomposition du Chi-2 et résidus")
                        
                        # Calculs
                        contribkhi2 = ((df_cont.values - res.expected_freq)**2) / res.expected_freq
                        frac_contrib = contribkhi2 / res.statistic
                        df_contrib = pd.DataFrame(frac_contrib, index=df_cont.index, columns=df_cont.columns)
                        
                        residu_std = (df_cont.values - res.expected_freq) / np.sqrt(res.expected_freq)
                        df_residu_std = pd.DataFrame(residu_std, index=df_cont.index, columns=df_cont.columns)
                        
                        chi2_col1, chi2_col2 = st.columns(2)
                        
                        with chi2_col1:
                            st.write("**Contribution au Chi-2 (fraction)**")
                            fig1, ax1 = plt.subplots(figsize=(10, 8))
                            sns.heatmap(df_contrib, annot=True, fmt='.3f', cmap=cmap_chi2,
                                    cbar=True, linewidths=0.5, ax=ax1)
                            ax1.set_title(title_chi2)
                            st.pyplot(fig1, use_container_width=True)
                        
                        with chi2_col2:
                            st.write("**Résidus standardisés**")
                            fig2, ax2 = plt.subplots(figsize=(10, 8))
                            sns.heatmap(df_residu_std, annot=True, fmt='.2f',
                                    cmap=cmap_residuals, center=0,
                                    cbar=True, linewidths=0.5, ax=ax2)
                            ax2.set_title(title_residuals)
                            st.pyplot(fig2, use_container_width=True)
            
            except Exception as e:
                st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                st.error("Vérifiez vos données et paramètres.")
        
        # Aide contextuelle
        with st.expander("ℹ️ Guide d'interprétation des résultats"):
            st.markdown("""
            ### 📊 **Interprétation des résultats AFC**
            
            **Test du Chi-2:**
            - **p-value < 0.05** : Les variables sont significativement associées
            - **p-value ≥ 0.05** : Pas d'association significative détectée
            
            **Graphiques des modalités:**
            - 🔴 **Points rouges** : modalités de la première variable
            - 🔵 **Points bleus** : modalités de la deuxième variable  
            - **Distance entre points** : Plus ils sont proches, plus ils sont associés
            - **Distance au centre** : Plus c'est éloigné, plus c'est caractéristique
            
            **Valeurs propres:**
            - Indiquent la qualité de représentation de chaque axe
            - Privilégier les premiers axes avec les valeurs les plus élevées
            
            **Contributions:**
            - Montrent l'importance de chaque modalité dans la construction des axes
            - Contribuions élevées = modalités qui "tirent" l'axe
            
            **Résidus standardisés:**
            - Valeurs > 2 ou < -2 : écarts significatifs à l'indépendance
            - Positifs : sur-représentation, Négatifs : sous-représentation
            """)
        
        st.success("🎉 Analyse AFC terminée ! Explorez les résultats dans les différents onglets.")
    def data_overview(df):
        st.header("Data Overview")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Number of Rows:** {df.shape[0]}")
            st.write(f"**Number of Columns:** {df.shape[1]}")

        with col2:
            st.write(f"**Missing Values:** {df.isna().sum().sum()}")
            st.write(f"**Duplicated Rows:** {df.duplicated().sum()}")

        # Data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10))

        # Column information
        st.subheader("Column Information")

        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Missing Values': df.isna().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info)

    
    def descriptive_statistics(df):
        st.header("Descriptive Statistics")

        # Select columns for analysis
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if numeric_cols:
            st.subheader("Numeric Variables")

            # Multiselect for columns
            selected_numeric = st.multiselect("Select numeric columns for analysis",
                                            numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])

            if selected_numeric:
                # Display descriptive stats for selected columns
                stats_df = df[selected_numeric].describe().T
                stats_df['median'] = df[selected_numeric].median()
                stats_df['skew'] = df[selected_numeric].skew()
                stats_df['kurtosis'] = df[selected_numeric].kurtosis()
                stats_df['missing'] = df[selected_numeric].isna().sum()
                stats_df['missing_percent'] = (df[selected_numeric].isna().sum() / len(df)) * 100

                st.dataframe(stats_df)

                # Option to download statistics
                stats_csv = stats_df.to_csv(index=True)
                st.download_button(
                    label="Download Statistics as CSV",
                    data=stats_csv,
                    file_name='descriptive_statistics.csv',
                    mime='text/csv',
                )

        if categorical_cols:
            st.subheader("Categorical Variables")

            # Multiselect for categorical columns
            selected_categorical = st.multiselect("Select categorical columns for analysis",
                                                categorical_cols,
                                                default=categorical_cols[:min(5, len(categorical_cols))])

            if selected_categorical:
                for col in selected_categorical:
                    st.write(f"**{col}**")

                    # Get value counts and percentages
                    value_counts = df[col].value_counts()
                    value_percent = df[col].value_counts(normalize=True) * 100

                    # Combine into a DataFrame
                    cat_stats = pd.DataFrame({
                        'Count': value_counts,
                        'Percentage (%)': value_percent
                    })

                    st.dataframe(cat_stats)

                    # Display simple bar chart
                    fig, ax = plt.subplots(figsize=(10, 5))
                    value_counts.plot(kind='bar', ax=ax)
                    ax.set_title(f"Frequency Distribution - {col}")
                    ax.set_ylabel("Count")
                    ax.set_xlabel(col)
                    plt.tight_layout()
                    st.pyplot(fig)


    def visualizations(df):
        """
        Fonction pour créer différents types de visualisations à partir d'un DataFrame
        avec personnalisation des couleurs et titres
        """
        st.header("📈 Visualisations")
        
        if df.empty:
            st.warning("Le DataFrame est vide. Aucune visualisation disponible.")
            return
        
        # Palette de couleurs prédéfinies
        color_palettes = {
            'Défaut': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'],
            'Viridis': px.colors.sequential.Viridis,
            'Plasma': px.colors.sequential.Plasma,
            'Blues': px.colors.sequential.Blues,
            'Reds': px.colors.sequential.Reds,
            'Greens': px.colors.sequential.Greens,
            'Pastel': px.colors.qualitative.Pastel,
            'Set1': px.colors.qualitative.Set1,
            'Set2': px.colors.qualitative.Set2,
            'Set3': px.colors.qualitative.Set3,
            'Dark2': px.colors.qualitative.Dark2,
            'Océan': ['#006994', '#13A5B7', '#26C9DE', '#B8E6F0'],
            'Sunset': ['#FF6B35', '#F7931E', '#FFD23F', '#06FFA5'],
            'Forest': ['#2D5016', '#4F7942', '#74A478', '#A8DADC']
        }
        
        # Correspondance pour les colorscales (graphiques continus)
        colorscale_mapping = {
            'Défaut': 'viridis',
            'Viridis': 'viridis',
            'Plasma': 'plasma',
            'Blues': 'blues',
            'Reds': 'reds',
            'Greens': 'greens',
            'Pastel': 'viridis',
            'Set1': 'viridis', 
            'Set2': 'viridis',
            'Set3': 'viridis',
            'Dark2': 'viridis',
            'Océan': 'teal',
            'Sunset': 'sunset',
            'Forest': 'greens'
        }
        
        # Section 1: Graphiques individuels
        st.subheader("Graphiques individuels")
        
        # Conteneurs pour organiser les contrôles
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_col = st.selectbox("Sélectionnez une colonne à visualiser:", options=df.columns)
        
        with col2:
            # Options de personnalisation dans un expander
            with st.expander("🎨 Personnalisation"):
                custom_title = st.text_input("Titre personnalisé (optionnel):", 
                                        placeholder=f"Graphique de {selected_col if selected_col else '...'}")
                color_palette = st.selectbox("Palette de couleurs:", options=list(color_palettes.keys()))
                custom_color = st.color_picker("Couleur personnalisée:", value="#FF6B6B")
                use_custom_color = st.checkbox("Utiliser couleur personnalisée")

        if selected_col and not df.empty:
            is_numeric = df[selected_col].dtype != 'object' and df[selected_col].dtype.name != 'category'
            unique_count = df[selected_col].nunique()
            
            # Déterminer les couleurs à utiliser
            if use_custom_color:
                colors = [custom_color]
            else:
                colors = color_palettes[color_palette]

            if is_numeric and unique_count > 10:  # Variable numérique continue
                graph_type = st.selectbox(
                    "Type de graphique pour variable numérique:",
                    options=["Histogramme", "Box Plot", "KDE (Density)", "Violin Plot", "Cumulative Distribution"],
                    key="num_graph_type"
                )
                
                # Titre par défaut ou personnalisé
                default_title = f"Histogramme de {selected_col}" if graph_type == "Histogramme" else f"{graph_type} de {selected_col}"
                title = custom_title if custom_title else default_title

                if graph_type == "Histogramme":
                    nbins = st.slider("Nombre de bins:", min_value=5, max_value=100, value=20)
                    fig = px.histogram(df, x=selected_col, nbins=nbins,
                                    title=title,
                                    color_discrete_sequence=colors,
                                    template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(marker_line_color='white', marker_line_width=1)
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "Box Plot":
                    fig = px.box(df, x=selected_col, title=title, 
                                points='all', color_discrete_sequence=colors,
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(marker=dict(color=colors[0] if colors else '#45B7D1', size=4, opacity=0.6))
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "KDE (Density)":
                    fig = px.density_contour(df, x=selected_col, title=title,
                                            template='plotly_white')
                    fig.update_traces(contours_coloring="fill", contours_showlabels=True,
                                    colorscale=colorscale_mapping[color_palette] if not use_custom_color else [[0, 'white'], [1, custom_color]])
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "Violin Plot":
                    fig = px.violin(df, y=selected_col, box=True, points='all',
                                    title=title,
                                    color_discrete_sequence=colors,
                                    template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(points_marker=dict(color=colors[0] if colors else '#F38BA8', size=3, opacity=0.7))
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "Cumulative Distribution":
                    fig = px.ecdf(df, x=selected_col, title=title,
                                color_discrete_sequence=colors,
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(line=dict(width=3))
                    st.plotly_chart(fig, use_container_width=True)

            else:  # Variable catégorielle
                graph_type = st.selectbox(
                    "Type de graphique pour variable catégorielle:",
                    options=["Bar Chart", "Pie Chart"],
                    key="cat_graph_type"
                )
                
                # Titre par défaut ou personnalisé
                default_title = f"{graph_type} de {selected_col}"
                title = custom_title if custom_title else default_title

                value_counts = df[selected_col].value_counts()

                if graph_type == "Bar Chart":
                    sort_option = st.checkbox("Trier par fréquence", value=True)
                    if sort_option:
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                    title=title,
                                    labels={'x': selected_col, 'y': 'Count'},
                                    color=value_counts.values,
                                    color_continuous_scale=colorscale_mapping[color_palette] if not use_custom_color else [[0, 'white'], [1, custom_color]],
                                    template='plotly_white')
                    else:
                        fig = px.bar(df, x=selected_col, title=title,
                                    color_discrete_sequence=colors,
                                    template='plotly_white')
                    
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5,
                        showlegend=False
                    )
                    fig.update_traces(marker_line_color='white', marker_line_width=1)
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "Pie Chart":
                    if len(value_counts) > 10:
                        st.warning(
                            f"La colonne a {len(value_counts)} valeurs uniques. Affichage limité aux 10 plus fréquentes.")
                        value_counts = value_counts.nlargest(10)

                    fig = px.pie(values=value_counts.values, names=value_counts.index,
                                title=title,
                                color_discrete_sequence=colors,
                                template='plotly_white')
                    fig.update_layout(
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
                        marker=dict(line=dict(color='white', width=2))
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with st.expander("Statistiques descriptives"):
                if is_numeric:
                    st.write(df[selected_col].describe())
                else:
                    st.write(f"Nombre de valeurs uniques: {df[selected_col].nunique()}")
                    st.write("Distribution des valeurs:")
                    st.write(df[selected_col].value_counts())
                    st.write("Distribution en pourcentage:")
                    st.write(df[selected_col].value_counts(normalize=True).mul(100).round(2).astype(str) + ' %')

        # Section 2: Graphiques bidimensionnels
        st.subheader("Graphiques bidimensionnels")
        
        # Organisation des contrôles
        control_cols = st.columns([1, 1, 1, 1])
        x_col = control_cols[0].selectbox(label='X_colonne', options=df.columns)
        y_col = control_cols[1].selectbox(label='Y_colonne', options=df.columns)
        
        # Personnalisation pour graphiques bidimensionnels
        with control_cols[3]:
            with st.expander("🎨 Style"):
                custom_title_2d = st.text_input("Titre personnalisé:", 
                                            placeholder=f"{y_col} vs {x_col}" if x_col and y_col else "...")
                color_palette_2d = st.selectbox("Palette:", options=list(color_palettes.keys()), key="palette_2d")
                custom_color_2d = st.color_picker("Couleur:", value="#FF6B6B", key="color_2d")
                use_custom_color_2d = st.checkbox("Couleur personnalisée", key="custom_2d")

        # Vérifier les types de données pour proposer des graphiques appropriés
        if x_col and y_col and not df.empty:
            x_is_object = df[x_col].dtype == 'object' or df[x_col].dtype.name == 'category' or df[x_col].nunique() <= 10
            y_is_object = df[y_col].dtype == 'object' or df[y_col].dtype.name == 'category' or df[y_col].nunique() <= 10
            
            # Déterminer les couleurs pour graphiques 2D
            if use_custom_color_2d:
                colors_2d = [custom_color_2d]
            else:
                colors_2d = color_palettes[color_palette_2d]

            # Proposer différents types de graphiques selon les types de données
            if x_is_object and y_is_object:
                # Les deux sont catégorielles
                graph_type = control_cols[2].selectbox(
                    label='Type de graphique',
                    options=['Bar Chart', 'Count Plot', 'Heatmap'],
                    key='cat_cat'
                )
                
                default_title_2d = f"{graph_type} of {x_col} by {y_col}"
                title_2d = custom_title_2d if custom_title_2d else default_title_2d

                if graph_type == 'Bar Chart':
                    fig = px.bar(df, x=x_col, color=y_col, title=title_2d,
                                color_discrete_sequence=colors_2d,
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(1,1,1,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Count Plot':
                    cross_tab = pd.crosstab(df[x_col], df[y_col])
                    fig = px.imshow(cross_tab, text_auto=True, aspect="auto",
                                    title=title_2d,
                                    color_continuous_scale=colorscale_mapping[color_palette_2d] if not use_custom_color_2d else [[0, 'white'], [1, custom_color_2d]],
                                    template='plotly_white')
                    fig.update_layout(
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{x}</b> & <b>%{y}</b><br>Count: %{z}<extra></extra>'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Heatmap':
                    cross_tab_norm = pd.crosstab(df[x_col], df[y_col], normalize='index')
                    fig = px.imshow(cross_tab_norm, text_auto='.1%', aspect="auto",
                                    color_continuous_scale=colorscale_mapping[color_palette_2d] if not use_custom_color_2d else [[0, 'white'], [1, custom_color_2d]],
                                    title=title_2d,
                                    template='plotly_white')
                    fig.update_layout(
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{x}</b> & <b>%{y}</b><br>Percentage: %{z:.1%}<extra></extra>'
                    )
                    st.plotly_chart(fig, use_container_width=True)

            elif (x_is_object and not y_is_object) or (not x_is_object and y_is_object):
                # Une variable catégorielle et une numérique
                cat_col = x_col if x_is_object else y_col
                num_col = y_col if x_is_object else x_col

                graph_type = control_cols[2].selectbox(
                    label='Type de graphique',
                    options=['Box Plot', 'Violin Plot', 'Bar Chart', 'Swarm Plot'],
                    key='cat_num'
                )
                
                default_title_2d = f"{graph_type} of {num_col} by {cat_col}"
                title_2d = custom_title_2d if custom_title_2d else default_title_2d

                if graph_type == 'Box Plot':
                    fig = px.box(df, x=cat_col, y=num_col,
                                title=title_2d,
                                color=cat_col,
                                color_discrete_sequence=colors_2d,
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Violin Plot':
                    fig = px.violin(df, x=cat_col, y=num_col, box=True,
                                    title=title_2d,
                                    color=cat_col,
                                    color_discrete_sequence=colors_2d,
                                    template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Bar Chart':
                    agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
                    fig = px.bar(agg_df, x=cat_col, y=num_col,
                                title=title_2d,
                                color=num_col,
                                color_continuous_scale=colorscale_mapping[color_palette_2d] if not use_custom_color_2d else [[0, 'white'], [1, custom_color_2d]],
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5,
                        showlegend=False
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{x}</b><br>Average: %{y:.2f}<extra></extra>',
                        marker_line_color='white',
                        marker_line_width=1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Swarm Plot':
                    fig = px.strip(df, x=cat_col, y=num_col,
                                title=title_2d,
                                color=cat_col,
                                color_discrete_sequence=colors_2d,
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5,
                        showlegend=False
                    )
                    fig.update_traces(marker=dict(size=6, opacity=0.7))
                    st.plotly_chart(fig, use_container_width=True)

            else:
                # Les deux sont numériques
                graph_type = control_cols[2].selectbox(
                    label='Type de graphique',
                    options=['Scatter Plot', 'Line Plot', 'Hexbin', 'Density Contour', 'Bubble Chart'],
                    key='num_num'
                )
                
                default_title_2d = f"{graph_type} of {y_col} vs {x_col}"
                title_2d = custom_title_2d if custom_title_2d else default_title_2d

                if graph_type == 'Scatter Plot':
                    fig = px.scatter(df, x=x_col, y=y_col,
                                    title=title_2d,
                                    color_discrete_sequence=colors_2d,
                                    template='plotly_white',
                                    opacity=0.7)
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Line Plot':
                    sorted_df = df.sort_values(by=x_col)
                    fig = px.line(sorted_df, x=x_col, y=y_col,
                                title=title_2d,
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(
                        line=dict(color=colors_2d[0] if colors_2d else '#4ECDC4', width=3),
                        marker=dict(size=6, color=colors_2d[0] if colors_2d else '#FF6B6B', line=dict(width=1, color='white'))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Hexbin':
                    fig = px.density_heatmap(df, x=x_col, y=y_col, nbinsx=20, nbinsy=20,
                                            title=title_2d,
                                            color_continuous_scale=colorscale_mapping[color_palette_2d] if not use_custom_color_2d else [[0, 'white'], [1, custom_color_2d]],
                                            template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Density Contour':
                    fig = px.density_contour(df, x=x_col, y=y_col,
                                            title=title_2d,
                                            template='plotly_white')
                    fig.update_traces(colorscale=colorscale_mapping[color_palette_2d] if not use_custom_color_2d else [[0, 'white'], [1, custom_color_2d]])
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif graph_type == 'Bubble Chart':
                    num_cols = df.select_dtypes(include=['float', 'int']).columns
                    if len(num_cols) > 2 and set([x_col, y_col]).issubset(set(num_cols)):
                        size_col = [col for col in num_cols if col not in [x_col, y_col]][0]
                        fig = px.scatter(df, x=x_col, y=y_col, size=size_col,
                                        title=title_2d,
                                        color_discrete_sequence=colors_2d,
                                        template='plotly_white')
                    else:
                        fig = px.scatter(df, x=x_col, y=y_col, size_max=15,
                                        title=title_2d,
                                        color_discrete_sequence=colors_2d,
                                        template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)                

    def correlation_analysis(df):
        st.header("Correlation Analysis")

        # Get numeric columns for correlation
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
            return

        # Select columns for analysis
        selected_cols = st.multiselect("Select columns for correlation analysis",
                                    numeric_cols,
                                    default=numeric_cols[:min(6, len(numeric_cols))])

        if len(selected_cols) < 2:
            st.warning("Please select at least 2 columns.")
            return

        # Choose correlation method
        corr_method = st.radio("Select correlation method",
                            ["Pearson", "Spearman", "Kendall"])

        # Calculate correlation
        corr_matrix = df[selected_cols].corr(method=corr_method.lower())

        # Display correlation table
        st.subheader("Correlation Matrix")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1))

        # Heatmap
        st.subheader("Correlation Heatmap")

        fig = plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                            fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
        plt.title(f'{corr_method} Correlation Heatmap')
        plt.tight_layout()
        st.pyplot(fig)

        # Pairwise correlation analysis
        st.subheader("Pairwise Correlation")

        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox("Select first variable", selected_cols)

        with col2:
            y_col = st.selectbox("Select second variable",
                                [col for col in selected_cols if col != x_col])

        # Scatter plot with regression line
        fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                        title=f"Correlation between {x_col} and {y_col}")

        # Calculate and display correlation coefficient
        corr_value = df[[x_col, y_col]].corr(method=corr_method.lower()).iloc[0, 1]
        fig.add_annotation(x=0.5, y=0.95,
                        text=f"{corr_method} correlation: {corr_value:.4f}",
                        showarrow=False,
                        font=dict(size=14),
                        xref="paper", yref="paper")

        st.plotly_chart(fig, use_container_width=True)


    def advanced_analysis(df):
        st.header("Advanced Analysis")

        analysis_type = st.selectbox("Select Analysis Type",
                                    ["Principal Component Analysis (PCA)",
                                    "Distribution Analysis",
                                    "Outlier Detection",
                                    "Time Series Decomposition"])

        if analysis_type == "Principal Component Analysis (PCA)":
            pca_analysis(df)

        elif analysis_type == "Distribution Analysis":
            distribution_analysis(df)

        elif analysis_type == "Outlier Detection":
            outlier_analysis(df)

        elif analysis_type == "Time Series Decomposition":
            time_series_analysis(df)


    def pca_analysis(df):
        st.subheader("Principal Component Analysis (PCA)")

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] < 2:
            st.warning("PCA requires at least 2 numeric columns.")
            return

        # Select columns for PCA
        selected_cols = st.multiselect("Select columns for PCA",
                                    numeric_df.columns.tolist(),
                                    default=numeric_df.columns.tolist()[:min(5, len(numeric_df.columns))])

        if len(selected_cols) < 2:
            st.warning("Please select at least 2 columns.")
            return

        # Get data for PCA
        X = numeric_df[selected_cols].copy()

        # Handle missing values
        if X.isna().any().any():
            st.warning("Missing values detected. They will be filled with column means.")
            X = X.fillna(X.mean())

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Number of components
        n_components = st.slider("Number of components", min_value=2,
                                max_value=min(len(selected_cols), 10), value=2)

        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        # Explained variance
        explained_variance = pca.explained_variance_ratio_ * 100

        # Display explained variance
        st.write("**Explained Variance by Component**")

        # Create DataFrame for explained variance
        exp_var_df = pd.DataFrame({
            'Component': [f'PC{i + 1}' for i in range(n_components)],
            'Explained Variance (%)': explained_variance,
            'Cumulative Variance (%)': np.cumsum(explained_variance)
        })

        st.dataframe(exp_var_df)

        # Plot explained variance
        fig = px.bar(exp_var_df, x='Component', y='Explained Variance (%)',
                    title="Explained Variance by Principal Component")

        fig.add_trace(go.Scatter(x=exp_var_df['Component'],
                                y=exp_var_df['Cumulative Variance (%)'],
                                mode='lines+markers', name='Cumulative Variance',
                                line=dict(color='red', width=2)))

        st.plotly_chart(fig, use_container_width=True)

        # Display PCA components
        st.write("**PCA Components Loading**")

        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i + 1}' for i in range(n_components)],
            index=selected_cols
        )

        st.dataframe(loadings)

        # Plot loadings
        fig = px.imshow(loadings,
                        labels=dict(x="Principal Components", y="Features"),
                        x=[f'PC{i + 1}' for i in range(n_components)],
                        y=selected_cols,
                        color_continuous_scale='RdBu_r',
                        title="PCA Component Loadings")

        st.plotly_chart(fig, use_container_width=True)

        # Plot PCA scatter plot for first two components
        if n_components >= 2:
            st.subheader("PCA Scatter Plot (First Two Components)")

            # Create DataFrame with PCA results
            pca_df = pd.DataFrame(
                X_pca[:, :2],
                columns=['PC1', 'PC2']
            )

            # Add color column if available
            color_col = None
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

            if categorical_cols:
                color_col = st.selectbox("Color points by (optional)",
                                        ["None"] + categorical_cols)

                if color_col != "None":
                    pca_df[color_col] = df[color_col].values

            # Create scatter plot
            if color_col and color_col != "None":
                fig = px.scatter(pca_df, x='PC1', y='PC2', color=color_col,
                                title=f"PCA Scatter Plot (PC1 vs PC2, colored by {color_col})")
            else:
                fig = px.scatter(pca_df, x='PC1', y='PC2',
                                title="PCA Scatter Plot (PC1 vs PC2)")

            # Add axis labels with explained variance
            fig.update_xaxes(title=f"PC1 ({explained_variance[0]:.2f}%)")
            fig.update_yaxes(title=f"PC2 ({explained_variance[1]:.2f}%)")

            # Add loading vectors
            if st.checkbox("Show feature loadings on scatter plot"):
                loadings_x = loadings['PC1'].values
                loadings_y = loadings['PC2'].values

                for i, feature in enumerate(selected_cols):
                    fig.add_shape(
                        type='line',
                        x0=0, y0=0,
                        x1=loadings_x[i] * 3,  # Scaling for visibility
                        y1=loadings_y[i] * 3,  # Scaling for visibility
                        line=dict(color='red', width=1, dash='dot')
                    )

                    fig.add_annotation(
                        x=loadings_x[i] * 3.5,  # Position text at end of arrow
                        y=loadings_y[i] * 3.5,
                        text=feature,
                        showarrow=False,
                        font=dict(color='red')
                    )

            st.plotly_chart(fig, use_container_width=True)


    def distribution_analysis(df):
        st.subheader("Distribution Analysis")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available for distribution analysis.")
            return

        # Select column for analysis
        col = st.selectbox("Select column for distribution analysis", numeric_cols)

        # Remove NaN values
        data = df[col].dropna()

        col1, col2 = st.columns(2)

        with col1:
            # Descriptive statistics
            st.write("**Descriptive Statistics**")
            stats = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
                'Value': [
                    data.mean(),
                    data.median(),
                    data.std(),
                    data.min(),
                    data.max(),
                    data.skew(),
                    data.kurtosis()
                ]
            })
            st.dataframe(stats)

        with col2:
            # Calculate quartiles and IQR
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            st.write("**Quantiles**")
            quantiles = pd.DataFrame({
                'Quantile': ['0% (Min)', '25%', '50% (Median)', '75%', '100% (Max)', 'IQR'],
                'Value': [
                    data.min(),
                    q1,
                    data.median(),
                    q3,
                    data.max(),
                    iqr
                ]
            })
            st.dataframe(quantiles)

        # Distribution plots
        st.write("**Distribution Visualization**")

        plot_type = st.radio("Select plot type",
                            ["Histogram with KDE", "Box Plot", "Violin Plot", "QQ Plot"])

        if plot_type == "Histogram with KDE":
            bin_count = st.slider("Number of bins", min_value=5, max_value=100, value=30)

            fig = plt.figure(figsize=(10, 6))
            sns.histplot(data, kde=True, bins=bin_count)
            plt.title(f"Histogram with KDE for {col}")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

        elif plot_type == "Box Plot":
            fig = plt.figure(figsize=(10, 6))
            sns.boxplot(x=data)
            plt.title(f"Box Plot for {col}")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

        elif plot_type == "Violin Plot":
            fig = plt.figure(figsize=(10, 6))
            sns.violinplot(x=data)
            plt.title(f"Violin Plot for {col}")
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)

        elif plot_type == "QQ Plot":
            from scipy import stats as scipy_stats

            fig = plt.figure(figsize=(10, 6))

            # Create Q-Q plot
            scipy_stats.probplot(data, dist="norm", plot=plt)
            plt.title(f"Q-Q Plot for {col}")
            plt.grid(True, alpha=0.3)

            st.pyplot(fig)

        # Additional distribution characteristics
        st.write("**Test for Normality**")

        from scipy import stats as scipy_stats

        # Perform Shapiro-Wilk test for normality
        if len(data) <= 5000:  # Shapiro-Wilk works best for smaller samples
            stat, p_value = scipy_stats.shapiro(data.sample(min(len(data), 5000)))
            test_name = "Shapiro-Wilk"
        else:
            # For larger datasets, use K-S test
            stat, p_value = scipy_stats.kstest(data, 'norm')
            test_name = "Kolmogorov-Smirnov"

        # Display test results
        normality_result = pd.DataFrame({
            'Test': [test_name],
            'Statistic': [stat],
            'p-value': [p_value],
            'Interpretation': [
                "Data likely follows a normal distribution" if p_value > 0.05 else
                "Data likely does not follow a normal distribution"
            ]
        })

        st.dataframe(normality_result)

        if p_value <= 0.05:
            st.info("Since the p-value is ≤ 0.05, we reject the null hypothesis that the data is normally distributed.")
        else:
            st.info(
                "Since the p-value is > 0.05, we cannot reject the null hypothesis that the data is normally distributed.")


    def outlier_analysis(df):
        st.subheader("Outlier Detection")

        # Get numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available for outlier detection.")
            return

        # Select column for analysis
        col = st.selectbox("Select column for outlier detection", numeric_cols)

        # Remove NaN values
        data = df[col].dropna()

        # Select outlier detection method
        method = st.radio("Select outlier detection method",
                        ["Z-Score", "IQR (Interquartile Range)", "Both"])

        # Z-Score Analysis
        if method in ["Z-Score", "Both"]:
            st.write("**Z-Score Method**")

            z_threshold = st.slider("Z-Score threshold",
                                    min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                                    key="z_score_slider")

            # Calculate Z-scores
            z_scores = (data - data.mean()) / data.std()
            outliers_z = data[abs(z_scores) > z_threshold]

            if not outliers_z.empty:
                st.write(f"Found {len(outliers_z)} outliers using Z-Score method (threshold: ±{z_threshold})")

                # Show outliers table
                outlier_df = pd.DataFrame({
                    'Index': outliers_z.index,
                    'Value': outliers_z.values,
                    'Z-Score': z_scores[outliers_z.index].values
                })

                st.dataframe(outlier_df)

                # Visualize outliers
                fig = px.scatter(x=data.index, y=data,
                                title=f"Z-Score Outliers for {col}")

                # Add outliers as different markers
                fig.add_scatter(x=outliers_z.index, y=outliers_z,
                                mode='markers', marker=dict(color='red', size=10),
                                name='Outliers')

                # Add threshold lines
                mean_val = data.mean()
                std_val = data.std()

                fig.add_hline(y=mean_val + z_threshold * std_val, line_dash="dash",
                            line_color="red", annotation_text=f"+{z_threshold} SD")
                fig.add_hline(y=mean_val - z_threshold * std_val, line_dash="dash",
                            line_color="red", annotation_text=f"-{z_threshold} SD")

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No outliers found using Z-Score method with threshold ±{z_threshold}")

        # IQR Analysis
        if method in ["IQR (Interquartile Range)", "Both"]:
            st.write("**IQR Method**")

            iqr_multiplier = st.slider("IQR multiplier",
                                    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                                    key="iqr_slider")

            # Calculate IQR bounds
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]

            if not outliers_iqr.empty:
                st.write(f"Found {len(outliers_iqr)} outliers using IQR method (multiplier: {iqr_multiplier})")

                # Show outliers table
                outlier_df = pd.DataFrame({
                    'Index': outliers_iqr.index,
                    'Value': outliers_iqr.values,
                    'Type': ['Below lower bound' if val < lower_bound else 'Above upper bound'
                            for val in outliers_iqr.values]
                })

                st.dataframe(outlier_df)

                # Visualize outliers
                fig = px.box(data, title=f"Box Plot with Outliers for {col}")

                # Add scatter points for outliers
                fig.add_scatter(x=[0] * len(outliers_iqr), y=outliers_iqr,
                                mode='markers', marker=dict(color='red', size=8),
                                name='Outliers')

                st.plotly_chart(fig, use_container_width=True)

                # Show outlier distribution
                fig = px.scatter(x=data.index, y=data,
                                title=f"IQR Outliers for {col}")

                # Add outliers as different markers
                fig.add_scatter(x=outliers_iqr.index, y=outliers_iqr,
                                mode='markers', marker=dict(color='red', size=10),
                                name='Outliers')

                # Add threshold lines
                fig.add_hline(y=upper_bound, line_dash="dash",
                            line_color="red", annotation_text=f"Upper bound ({upper_bound:.2f})")
                fig.add_hline(y=lower_bound, line_dash="dash",
                            line_color="red", annotation_text=f"Lower bound ({lower_bound:.2f})")

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No outliers found using IQR method with multiplier {iqr_multiplier}")

        # Provide options for handling outliers
        if method == "Both" and (not outliers_z.empty or not outliers_iqr.empty):
            st.subheader("Outlier Comparison")

            if not outliers_z.empty and not outliers_iqr.empty:
                # Find common outliers
                common_outliers = set(outliers_z.index).intersection(set(outliers_iqr.index))

                st.write(f"**Common outliers detected by both methods:** {len(common_outliers)}")

                # Show Venn diagram
                from matplotlib_venn import venn2

                fig, ax = plt.subplots(figsize=(8, 6))
                venn = venn2([set(outliers_z.index), set(outliers_iqr.index)],
                            set_labels=('Z-Score', 'IQR'))
                plt.title("Outlier Detection Comparison")
                st.pyplot(fig)

        # Add options for handling outliers
        if st.checkbox("Show outlier handling options"):
            st.write("**Outlier Handling Options**")

            handling_method = st.radio(
                "Select method to handle outliers",
                ["Remove outliers", "Cap outliers", "Replace with mean/median", "No action"]
            )

            # Function to get the combined outliers from both methods
            def get_combined_outliers():
                if method == "Z-Score":
                    return outliers_z.index
                elif method == "IQR (Interquartile Range)":
                    return outliers_iqr.index
                else:  # Both
                    return list(set(outliers_z.index) | set(outliers_iqr.index))

            if handling_method != "No action":
                outlier_indices = get_combined_outliers()

                if len(outlier_indices) == 0:
                    st.info("No outliers to handle.")
                else:
                    # Create a copy of the dataframe for demonstration
                    modified_df = df.copy()

                    if handling_method == "Remove outliers":
                        modified_df = modified_df.drop(outlier_indices)
                        action_text = "removed"

                    elif handling_method == "Cap outliers":
                        if method in ["IQR (Interquartile Range)", "Both"]:
                            # Cap using IQR bounds
                            modified_df.loc[modified_df[modified_df[col] > upper_bound].index, col] = upper_bound
                            modified_df.loc[modified_df[modified_df[col] < lower_bound].index, col] = lower_bound
                        else:
                            # Cap using Z-score
                            mean_val = data.mean()
                            std_val = data.std()
                            modified_df.loc[modified_df[modified_df[
                                                            col] > mean_val + z_threshold * std_val].index, col] = mean_val + z_threshold * std_val
                            modified_df.loc[modified_df[modified_df[
                                                            col] < mean_val - z_threshold * std_val].index, col] = mean_val - z_threshold * std_val

                        action_text = "capped"

                    elif handling_method == "Replace with mean/median":
                        replacement = st.radio("Replace with:", ["Mean", "Median"])

                        if replacement == "Mean":
                            replacement_value = data.mean()
                        else:  # Median
                            replacement_value = data.median()

                        modified_df.loc[outlier_indices, col] = replacement_value
                        action_text = f"replaced with {replacement.lower()}"

                    st.write(f"**Result after outliers {action_text}:**")

                    # Show before and after histograms
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("Before:")
                        fig = px.histogram(df[col].dropna(), nbins=30,
                                        title=f"Original Distribution of {col}")
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.write("After:")
                        fig = px.histogram(modified_df[col].dropna(), nbins=30,
                                        title=f"Modified Distribution of {col}")
                        st.plotly_chart(fig, use_container_width=True)

                    # Show summary statistics
                    st.write("**Summary Statistics Comparison:**")

                    summary_comparison = pd.DataFrame({
                        'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness'],
                        'Before': [
                            df[col].dropna().count(),
                            df[col].dropna().mean(),
                            df[col].dropna().median(),
                            df[col].dropna().std(),
                            df[col].dropna().min(),
                            df[col].dropna().max(),
                            df[col].dropna().skew()
                        ],
                        'After': [
                            modified_df[col].dropna().count(),
                            modified_df[col].dropna().mean(),
                            modified_df[col].dropna().median(),
                            modified_df[col].dropna().std(),
                            modified_df[col].dropna().min(),
                            modified_df[col].dropna().max(),
                            modified_df[col].dropna().skew()
                        ]
                    })

                    st.dataframe(summary_comparison)


    def time_series_analysis(df):
        st.subheader("Time Series Analysis")

        # Check if there are datetime columns
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            # Try to convert to datetime
            elif df[col].dtypes == 'object':
                try:
                    pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except:
                    pass

        if not datetime_cols:
            st.warning("No datetime columns detected. Time series analysis requires a datetime column.")
            return

        # Select datetime column
        date_col = st.selectbox("Select date/time column", datetime_cols)

        # Make sure it's datetime format
        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except:
            st.error(f"Failed to convert {date_col} to datetime format.")
            return

        # Select value column
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available for time series analysis.")
            return

        value_col = st.selectbox("Select value column to analyze", numeric_cols)

        # Time resampling
        freq_options = {
            "Original (no resampling)": None,
            "Daily": "D",
            "Weekly": "W",
            "Monthly": "M",
            "Quarterly": "Q",
            "Yearly": "Y"
        }

        resample_freq = st.selectbox("Select time aggregation", list(freq_options.keys()))

        # Aggregation method
        agg_method = st.selectbox("Select aggregation method",
                                ["Mean", "Sum", "Min", "Max", "Median", "Count"])

        # Prepare time series data
        try:
            # Sort by date
            ts_df = df[[date_col, value_col]].sort_values(by=date_col)

            # Set date as index
            ts_df = ts_df.set_index(date_col)

            # Resample if selected
            if freq_options[resample_freq] is not None:
                if agg_method == "Mean":
                    ts_df = ts_df.resample(freq_options[resample_freq]).mean()
                elif agg_method == "Sum":
                    ts_df = ts_df.resample(freq_options[resample_freq]).sum()
                elif agg_method == "Min":
                    ts_df = ts_df.resample(freq_options[resample_freq]).min()
                elif agg_method == "Max":
                    ts_df = ts_df.resample(freq_options[resample_freq]).max()
                elif agg_method == "Median":
                    ts_df = ts_df.resample(freq_options[resample_freq]).median()
                else:  # Count
                    ts_df = ts_df.resample(freq_options[resample_freq]).count()

            # Reset index for plotting
            ts_df = ts_df.reset_index()

            # Basic time series plot
            st.write("**Time Series Plot**")

            # Interactive plot with Plotly
            fig = px.line(ts_df, x=date_col, y=value_col,
                        title=f"Time Series: {value_col} over time")

            # Add range slider
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )

            st.plotly_chart(fig, use_container_width=True)

            # Time series decomposition
            if st.checkbox("Show time series decomposition"):
                from statsmodels.tsa.seasonal import seasonal_decompose

                # Set date as index again for decomposition
                ts_data = ts_df.set_index(date_col)[value_col]

                if len(ts_data) < 4:
                    st.warning("Not enough data points for decomposition. Need at least 4 points.")
                else:
                    # Determine appropriate period for decomposition
                    if freq_options[resample_freq] is None:
                        # If using original data, try to infer frequency
                        period_options = {
                            "Auto-detect": None,
                            "Daily (7)": 7,
                            "Weekly (52)": 52,
                            "Monthly (12)": 12,
                            "Quarterly (4)": 4
                        }
                    else:
                        # For resampled data
                        if freq_options[resample_freq] == "D":
                            period_options = {
                                "Weekly (7)": 7,
                                "Bi-weekly (14)": 14,
                                "Monthly (30)": 30,
                                "Custom": "custom"
                            }
                        elif freq_options[resample_freq] == "W":
                            period_options = {
                                "Monthly (4)": 4,
                                "Quarterly (13)": 13,
                                "Yearly (52)": 52,
                                "Custom": "custom"
                            }
                        elif freq_options[resample_freq] == "M":
                            period_options = {
                                "Quarterly (3)": 3,
                                "Yearly (12)": 12,
                                "Custom": "custom"
                            }
                        elif freq_options[resample_freq] in ["Q", "Y"]:
                            period_options = {
                                "Yearly (4)": 4,
                                "Custom": "custom"
                            }
                        else:
                            period_options = {
                                "Auto-detect": None,
                                "Custom": "custom"
                            }

                    period_selection = st.selectbox("Select seasonality period", list(period_options.keys()))

                    if period_selection == "Custom":
                        period = st.number_input("Enter custom period", min_value=2, max_value=len(ts_data) // 2, value=12)
                    else:
                        period = period_options[period_selection]

                        # Auto-detect if selected
                        if period is None:
                            # Simple heuristic for period detection
                            if len(ts_data) >= 730:  # At least 2 years of daily data
                                period = 365  # Annual cycle for daily data
                            elif len(ts_data) >= 60:  # At least 2 years of monthly data
                                period = 12  # Annual cycle for monthly data
                            elif len(ts_data) >= 16:  # At least 2 years of quarterly data
                                period = 4  # Annual cycle for quarterly data
                            elif len(ts_data) >= 14:  # At least 2 weeks of daily data
                                period = 7  # Weekly cycle
                            else:
                                period = 4  # Default fallback

                    # Make sure period is not too large compared to data length
                    period = min(period, len(ts_data) // 2)

                    try:
                        # Perform decomposition
                        decomposition = seasonal_decompose(ts_data, model='additive', period=period)

                        # Create DataFrames for the components
                        trend = decomposition.trend
                        seasonal = decomposition.seasonal
                        residual = decomposition.resid

                        # Plot components
                        st.write(f"**Time Series Decomposition (Period: {period})**")

                        fig = plt.figure(figsize=(12, 10))
                        plt.subplot(411)
                        plt.plot(ts_data, label='Original')
                        plt.legend(loc='upper left')
                        plt.title('Time Series Decomposition')

                        plt.subplot(412)
                        plt.plot(trend, label='Trend')
                        plt.legend(loc='upper left')

                        plt.subplot(413)
                        plt.plot(seasonal, label='Seasonality')
                        plt.legend(loc='upper left')

                        plt.subplot(414)
                        plt.plot(residual, label='Residuals')
                        plt.legend(loc='upper left')

                        plt.tight_layout()
                        st.pyplot(fig)

                        # Interactive decomposition with Plotly
                        decomp_df = pd.DataFrame({
                            'Date': ts_data.index,
                            'Original': ts_data.values,
                            'Trend': trend.values,
                            'Seasonal': seasonal.values,
                            'Residual': residual.values
                        }).dropna()

                        # Plot original and trend
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Original'],
                                                mode='lines', name='Original'))
                        fig.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Trend'],
                                                mode='lines', name='Trend', line=dict(width=3)))

                        fig.update_layout(title='Original Time Series and Trend',
                                        xaxis_title='Date',
                                        yaxis_title='Value')

                        st.plotly_chart(fig, use_container_width=True)

                        # Plot seasonality
                        fig = px.line(decomp_df, x='Date', y='Seasonal',
                                    title='Seasonal Component')
                        st.plotly_chart(fig, use_container_width=True)

                        # Plot residuals
                        fig = px.scatter(decomp_df, x='Date', y='Residual',
                                        title='Residual Component')

                        # Add zero line
                        fig.add_hline(y=0, line_dash="dash", line_color="red")

                        st.plotly_chart(fig, use_container_width=True)

                        # Seasonality patterns (only for appropriate frequencies)
                        if period >= 4 and period <= 52:
                            # Group by period component to show seasonality pattern
                            seasonal_pattern = seasonal.groupby(seasonal.index.month if period == 12 else
                                                                seasonal.index.dayofweek if period == 7 else
                                                                seasonal.index % period).mean()

                            fig = px.line(x=seasonal_pattern.index, y=seasonal_pattern.values,
                                        title='Seasonal Pattern',
                                        labels={'x': 'Period Component', 'y': 'Average Effect'})

                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Decomposition error: {e}")
                        st.info("Try adjusting the period or using a different aggregation level.")

            # Show autocorrelation/partial autocorrelation
            if st.checkbox("Show autocorrelation analysis"):
                from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

                # Set date as index again
                ts_data = ts_df.set_index(date_col)[value_col]

                # Remove missing values
                ts_data = ts_data.dropna()

                if len(ts_data) < 3:
                    st.warning("Not enough data points for autocorrelation analysis.")
                else:
                    lags = st.slider("Number of lags", min_value=5, max_value=min(50, len(ts_data) - 1), value=20)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Autocorrelation Function (ACF)**")
                        fig = plt.figure(figsize=(10, 6))
                        plot_acf(ts_data, lags=lags, alpha=0.05, ax=plt.gca())
                        plt.title("Autocorrelation Function")
                        st.pyplot(fig)

                    with col2:
                        st.write("**Partial Autocorrelation Function (PACF)**")
                        fig = plt.figure(figsize=(10, 6))
                        plot_pacf(ts_data, lags=lags, alpha=0.05, ax=plt.gca())
                        plt.title("Partial Autocorrelation Function")
                        st.pyplot(fig)

                    st.info("**Interpretation:** Significant spikes in ACF indicate seasonality patterns. "
                            "PACF helps identify the order of an autoregressive model. "
                            "Spikes crossing the blue lines are statistically significant.")

        except Exception as e:
            st.error(f"Error in time series analysis: {e}")


    # Run the app
    if __name__ == "__main__":
        main()

def main_app():
    """Application principale avec admin intégré"""
    # Header avec info utilisateur
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Data Analysis")
    with col2:
        st.write(f"👤 {st.session_state.get('user_email', '')}")
        if st.button("🚪 Déconnexion", type="secondary"):
            logout_session(st.session_state.get('session_id'))
            st.session_state.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Onglets principaux
    if st.session_state.get('user_email') == 'ahmed.djalil.2004@gmail․com':
        # Admin a accès aux deux onglets
        main_tab1, main_tab2 = st.tabs(["📊 ADD", "🔧 Administration"])
        
        with main_tab1:
            add()
        
        with main_tab2:
            admin_panel()
    else:
        # Utilisateur standard n'a accès qu'à la visualisation
        add()

# ============ POINT D'ENTRÉE PRINCIPAL SÉCURISÉ ============

def main1():
    """Point d'entrée principal avec sécurité renforcée"""
    
    # Afficher la sidebar sécurisée
    secure_admin_sidebar()
    
    # Initialiser la base de données
    init_database()
    
    # Vérifier l'authentification
    session_id = st.session_state.get('session_id')
    is_valid, email = validate_session(session_id)
    
    if is_valid:
        # Utilisateur authentifié
        st.session_state['user_email'] = email
        main_app()
    else:
        # Utilisateur non authentifié
        if 'session_id' in st.session_state:
            del st.session_state['session_id']
        if 'user_email' in st.session_state:
            del st.session_state['user_email']
        login_interface()

# Exécuter l'application
if __name__ == "__main__":
    main1()
