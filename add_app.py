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
                    "🔥 Correlation Analysis", "📉 Advanced Analysis", "🔄 AFC","ACM","ACP"])

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
                    AFC_analysis(df)
                with tabs[6]:
                    ACM_analysis(df)
                with tabs[7]:
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
                                "🔥 Correlation Analysis", "📉 Advanced Analysis", "🔄 AFC","ACM","ACP"])

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
                    AFC_analysis(df)
                with tabs[6]:
                    ACM_analysis(df)
                with tabs[7]:
                    ACP_analysis(df)

    def ACP_analysis(df):
        """
        Fonction d'Analyse en Composantes Principales avec biplot amélioré
        """
        import warnings
        warnings.filterwarnings('ignore')
        
        # Vérifier si la bibliothèque fanalysis est disponible
        try:
            from fanalysis.pca import PCA
        except ImportError:
            st.error("❌ La bibliothèque 'fanalysis' n'est pas installée. Installez-la avec: pip install fanalysis")
            st.info("💡 Alternative: Vous pouvez utiliser sklearn.decomposition.PCA pour une version simplifiée de l'ACP")
            return
        
        from scipy.stats import pearsonr
        from sklearn.preprocessing import StandardScaler
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        
        st.header("🔍 Analyse en Composantes Principales (ACP)")
        st.markdown("Analysez les relations entre variables quantitatives et réduisez la dimensionnalité")
        
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
                ["Supprimer les lignes", "Remplacer par moyenne", "Remplacer par médiane", "Ne rien faire"],
                key="acp_missing_method",
                help="Choisissez comment traiter les valeurs manquantes"
            )
            
            # Standardisation
            standardize = st.checkbox(
                "Standardiser les données",
                value=True,
                key="acp_standardize",
                help="Recommandé quand les variables ont des unités différentes"
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
        elif missing_method == "Remplacer par moyenne":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
        elif missing_method == "Remplacer par médiane":
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())
        
        # Sélection des variables
        st.markdown("---")
        st.subheader("📋 Sélection des variables")
        
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
                help="Sélectionnez les variables numériques"
            )
        
        with col2:
            if vars_selected:
                st.write("**Variables sélectionnées:**")
                for var in vars_selected:
                    min_val = df_processed[var].min()
                    max_val = df_processed[var].max()
                    st.write(f"• {var}: {min_val:.2f} à {max_val:.2f}")
        
        if len(vars_selected) < 2:
            st.warning("⚠️ Veuillez sélectionner au moins 2 variables")
            return
        
        # Créer le dataset final
        X = df_processed[vars_selected].copy()
        
        # Vérifier qu'il n'y a que des valeurs numériques
        if not X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]:
            st.error("❌ Toutes les variables sélectionnées doivent être numériques")
            return
        
        # Standardisation
        if standardize:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(
                scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            X = X_scaled
            st.info("✅ Données standardisées")
        
        # Paramètres
        col1, col2 = st.columns(2)
        with col1:
            n_components = st.slider("Nombre de composantes", 2, min(10, len(vars_selected)), min(5, len(vars_selected)), key="acp_n_components")
        with col2:
            fig_size = st.slider("Taille des graphiques", 6, 12, 8, key="acp_fig_size")
        
        st.markdown("---")
        
        # Bouton pour lancer l'analyse
        if st.button("🚀 Lancer l'Analyse ACP", type="primary", key="acp_launch_button"):
            
            # Affichage des données finales
            st.subheader("✅ Données préparées")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Aperçu final:**")
                st.dataframe(X.head())
            
            with col2:
                st.write("**Statistiques descriptives:**")
                desc_stats = X.describe().round(3)
                st.dataframe(desc_stats)
            
            # Tests statistiques préliminaires
            st.markdown("---")
            st.subheader("📈 Analyse des corrélations")
            
            # Matrice de corrélation
            correlation_matrix = X.corr()
            
            # Visualisation de la matrice de corrélation
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, 
                    mask=mask,
                    annot=True, 
                    cmap='coolwarm', 
                    center=0,
                    square=True,
                    ax=ax)
            plt.title("Matrice de corrélation", fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Analyse de la corrélation moyenne
            mean_correlation = correlation_matrix.abs().mean().mean()
            strong_correlations = (correlation_matrix.abs() > 0.7).sum().sum() - len(vars_selected)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Corrélation moyenne", f"{mean_correlation:.3f}")
            with col2:
                st.metric("Corrélations fortes (>0.7)", strong_correlations)
            
            # Test d'adéquation pour l'ACP
            st.markdown("---")
            st.subheader("🧪 Adéquation des données pour l'ACP")
            
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
            
            # Exécution de l'ACP
            st.markdown("---")
            st.subheader("🎯 Résultats de l'ACP")
            
            try:
                # Exécution ACP
                acp = PCA(row_labels=X.index.values, col_labels=X.columns.values, n_components=n_components)
                acp.fit(X.values)
                
                # Valeurs propres
                st.subheader("📊 Valeurs propres et variance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_eigenvalues = acp.plot_eigenvalues()
                    plt.title("Valeurs propres", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig_eigenvalues)
                
                with col2:
                    fig_cumulative = acp.plot_eigenvalues("cumulative")
                    plt.title("Variance cumulée", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig_cumulative)
                
                # Tableau des valeurs propres
                eigenvalues = acp.eig_[1]
                variance_explained = (eigenvalues / eigenvalues.sum()) * 100
                cumulative_variance = np.cumsum(variance_explained)
                
                eigenvalues_df = pd.DataFrame({
                    'Composante': [f'PC{i+1}' for i in range(len(eigenvalues))],
                    'Valeur propre': eigenvalues.round(4),
                    'Variance (%)': variance_explained.round(2),
                    'Variance cumulée (%)': cumulative_variance.round(2)
                })
                
                st.write("**Tableau des valeurs propres:**")
                st.dataframe(eigenvalues_df, use_container_width=True)
                
                # Critère de Kaiser
                kaiser_threshold = 1.0
                significant_kaiser = eigenvalues > kaiser_threshold
                st.info(f"Critère de Kaiser (valeur propre > 1): {sum(significant_kaiser)} composantes significatives")
                
                # Critère du coude
                variance_80 = sum(cumulative_variance <= 80)
                st.info(f"Nombre de composantes pour 80% de variance: {variance_80}")
                
                # Sélection des axes pour visualisation
                st.markdown("---")
                st.subheader("📈 Visualisations")
                
                col1, col2 = st.columns(2)
                with col1:
                    axis_x = st.selectbox("Axe X", [f"PC{i+1}" for i in range(n_components)], index=0, key="acp_axis_x")
                    num_x_axis = int(axis_x[2:])
                with col2:
                    axis_y = st.selectbox("Axe Y", [f"PC{i+1}" for i in range(n_components)], index=1, key="acp_axis_y")
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
                    fig_cols = acp.mapping_col(num_x_axis=num_x_axis, num_y_axis=num_y_axis,
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
        try:
            from fanalysis.ca import CA
            fanalysis_available = True
        except ImportError:
            fanalysis_available = False
            st.warning("📦 Le package 'fanalysis' n'est pas installé. Certaines fonctionnalités ne seront pas disponibles.")
        
        # Afficher les premières lignes du dataset
        st.write("Aperçu des données:")
        st.dataframe(df.head())
        
        # Sélection des variables uniquement si des données sont chargées
        if not df.empty:
            # Filtrer les colonnes catégorielles et numériques
            categorical_cols = df.select_dtypes(exclude=['float']).columns.tolist()
            all_cols = df.columns.tolist()
            
            if categorical_cols:
                var1 = st.selectbox("Sélectionner la première variable (catégorielle)", categorical_cols)
                var2 = st.selectbox("Sélectionner la deuxième variable", all_cols)
                
                # Vérifier que les variables sont valides
                if var1 and var2 and var1 in df.columns and var2 in df.columns:
                    # Bouton pour exécuter l'analyse
                    run_analysis = st.button("Exécuter l'analyse AFC", type="primary")
                    
                    if run_analysis:
                        with st.spinner("Analyse en cours..."):
                            # Créer le tableau de contingence
                            df_cont = pd.crosstab(df[var1], df[var2])
                            
                            # Vérifier si le tableau de contingence est valide
                            if df_cont.size > 1:
                                st.subheader("Tableau de contingence:")
                                st.dataframe(df_cont)
                                
                                # Calculer les profils ligne et colonne
                                profil_ligne = df_cont.divide(df_cont.sum(axis=1), axis=0)
                                profil_colonne = df_cont.divide(df_cont.sum(axis=0), axis=1)
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("Profil ligne:")
                                    st.dataframe(profil_ligne)
                                with col2:
                                    st.subheader("Profil colonne:")
                                    st.dataframe(profil_colonne)
                                
                                # Calculer le test du chi-2
                                res = chi2_contingency(df_cont.values, correction=False)
                                
                                st.subheader("Test du Chi-2")
                                chi2_col1, chi2_col2 = st.columns(2)
                                with chi2_col1:
                                    st.metric("Valeur du Chi-2", f"{res.statistic:.4f}")
                                with chi2_col2:
                                    st.metric("p-value", f"{res.pvalue:.6f}")
                                
                                if res.pvalue < 0.05:
                                    st.success("✅ Les variables sont significativement dépendantes (p-value < 0.05)")
                                else:
                                    st.warning("⚠️ Les variables ne semblent pas dépendantes (p-value >= 0.05)")
                                
                                # Suite de l'analyse AFC seulement si fanalysis est disponible
                                if fanalysis_available:
                                    # Entrainement de l'algorithme AFC
                                    try:
                                        afc = CA(row_labels=df_cont.index, col_labels=df_cont.columns, stats=True)
                                        afc.fit(df_cont.values)
                                        
                                        # Valeurs propres
                                        st.subheader("Valeurs propres")
                                        fig_eigenvalues = afc.plot_eigenvalues()
                                        st.pyplot(fig_eigenvalues)
                                        
                                        # Créer des onglets pour l'affichage des résultats
                                        tab1, tab2, tab3 = st.tabs(["Analyse des modalités", "Contributions", "Résidus et Chi-2"])
                                        
                                        with tab1:
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.subheader("Analyse des modalités lignes")
                                                row_graph = afc.mapping_row(num_x_axis=1, num_y_axis=2)
                                                st.pyplot(row_graph)
                                            
                                            with col2:
                                                st.subheader("Analyse des modalités colonnes")
                                                col_graph = afc.mapping_col(num_x_axis=1, num_y_axis=2)
                                                st.pyplot(col_graph)
                                            
                                            st.subheader("Association ligne-colonne")
                                            st.pyplot(afc.mapping(num_x_axis=1, num_y_axis=2))
                                        
                                        with tab2:
                                            st.subheader("Contributions des modalités")
                                            contrib_col1, contrib_col2 = st.columns(2)
                                            
                                            with contrib_col1:
                                                st.write("Contributions des modalités lignes - Axe 1")
                                                st.pyplot(afc.plot_row_contrib(num_axis=1))
                                                
                                                st.write("Contributions des modalités colonnes - Axe 1")
                                                st.pyplot(afc.plot_col_contrib(num_axis=1))
                                            
                                            with contrib_col2:
                                                st.write("Contributions des modalités lignes - Axe 2")
                                                st.pyplot(afc.plot_row_contrib(num_axis=2))
                                                
                                                st.write("Contributions des modalités colonnes - Axe 2")
                                                st.pyplot(afc.plot_col_contrib(num_axis=2))
                                        
                                        with tab3:
                                            st.subheader("Décomposition du Chi-2")
                                            
                                            # Calcul des contributions au chi-2
                                            contribkhi2 = ((df_cont.values - res.expected_freq)**2)/res.expected_freq
                                            frac_contrib = contribkhi2/res.statistic 
                                            df_contrib = pd.DataFrame(frac_contrib, index=df_cont.index, columns=df_cont.columns)
                                            
                                            # Résidus standardisés
                                            residu_std = (df_cont.values - res.expected_freq) / np.sqrt(res.expected_freq)
                                            df_residu_std = pd.DataFrame(residu_std, index=df_cont.index, columns=df_cont.columns)
                                            
                                            chi2_col1, chi2_col2 = st.columns(2)
                                            with chi2_col1:
                                                st.write("Contribution au Chi-2 (fraction)")
                                                fig1, ax1 = plt.subplots(figsize=(10, 8))
                                                sns.heatmap(df_contrib, annot=True, fmt='.2f', cmap='Blues', 
                                                        cbar=True, linewidths=0.5, ax=ax1)
                                                ax1.set_title("Contribution au Chi-2 (fraction)")
                                                st.pyplot(fig1)
                                            
                                            with chi2_col2:
                                                st.write("Résidus standardisés")
                                                fig2, ax2 = plt.subplots(figsize=(10, 8))
                                                sns.heatmap(df_residu_std, annot=True, fmt='.2f', 
                                                        cmap=sns.diverging_palette(10, 240), 
                                                        cbar=True, linewidths=0.5, ax=ax2)
                                                ax2.set_title("Résidus standardisés")
                                                st.pyplot(fig2)
                                    
                                    except Exception as e:
                                        st.error(f"Erreur lors de l'analyse AFC: {e}")
                                else:
                                    st.warning("L'analyse AFC complète nécessite le package 'fanalysis'. " 
                                            "Installez-le avec: `pip install fanalysis`")
                            else:
                                st.error("Le tableau de contingence est vide ou invalide. Vérifiez vos sélections de variables.")
                else:
                    st.warning("Veuillez sélectionner deux variables différentes pour l'analyse.")
            else:
                st.warning("Aucune variable catégorielle détectée dans le jeu de données.")


    # Ajouter des informations d'aide en bas de page
    with st.expander("À propos de l'Analyse Factorielle des Correspondances"):
        st.write("""
        L'Analyse Factorielle des Correspondances (AFC) est une méthode statistique qui permet d'étudier 
        l'association entre deux variables qualitatives. Elle produit une représentation graphique qui 
        facilite l'interprétation des relations entre les modalités des variables.
        
        **Comment utiliser cet outil:**
        1. Chargez un fichier de données (CSV ou Excel)
        2. Sélectionnez deux variables à analyser
        3. Cliquez sur "Exécuter l'analyse AFC"
        4. Explorez les résultats dans les différents onglets
        
        **Interprétation des résultats:**
        - Les modalités proches sur le graphique ont des profils similaires
        - Les axes représentent les principales dimensions de variabilité des données
        - Les contributions indiquent l'importance de chaque modalité dans la construction des axes
        - Les résidus standardisés montrent les écarts à l'indépendance
        """)

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
        """
        st.header("📈 Visualisations")
        
        if df.empty:
            st.warning("Le DataFrame est vide. Aucune visualisation disponible.")
            return
        
        # Section 1: Graphiques individuels
        st.subheader("Graphiques individuels")
        selected_col = st.selectbox("Sélectionnez une colonne à visualiser:", options=df.columns)

        if selected_col and not df.empty:
            is_numeric = df[selected_col].dtype != 'object' and df[selected_col].dtype.name != 'category'
            unique_count = df[selected_col].nunique()

            if is_numeric and unique_count > 10:  # Heuristic: more than 10 unique values, treat as continuous
                graph_type = st.selectbox(
                    "Type de graphique pour variable numérique:",
                    options=["Histogramme", "Box Plot", "KDE (Density)", "Violin Plot", "Cumulative Distribution"],
                    key="num_graph_type"
                )

                if graph_type == "Histogramme":
                    nbins = st.slider("Nombre de bins:", min_value=5, max_value=100, value=20)
                    fig = px.histogram(df, x=selected_col, nbins=nbins,
                                    title=f"Histogramme de {selected_col}",
                                    color_discrete_sequence=['#FF6B6B'],
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
                    fig = px.box(df, x=selected_col, title=f"Box Plot de {selected_col}", 
                                points='all', color_discrete_sequence=['#4ECDC4'],
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(marker=dict(color='#45B7D1', size=4, opacity=0.6))
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "KDE (Density)":
                    fig = px.density_contour(df, x=selected_col, title=f"KDE de {selected_col}",
                                            template='plotly_white')
                    fig.update_traces(contours_coloring="fill", contours_showlabels=True,
                                    colorscale='Viridis')
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
                                    title=f"Violin Plot de {selected_col}",
                                    color_discrete_sequence=['#96CEB4'],
                                    template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(points_marker=dict(color='#F38BA8', size=3, opacity=0.7))
                    st.plotly_chart(fig, use_container_width=True)

                elif graph_type == "Cumulative Distribution":
                    fig = px.ecdf(df, x=selected_col, title=f"Distribution cumulative de {selected_col}",
                                color_discrete_sequence=['#A8E6CF'],
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

            else:  # Treat as categorical
                graph_type = st.selectbox(
                    "Type de graphique pour variable catégorielle:",
                    options=["Bar Chart", "Pie Chart"],
                    key="cat_graph_type"
                )

                value_counts = df[selected_col].value_counts()

                if graph_type == "Bar Chart":
                    sort_option = st.checkbox("Trier par fréquence", value=True)
                    if sort_option:
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                    title=f"Bar Chart de {selected_col}",
                                    labels={'x': selected_col, 'y': 'Count'},
                                    color=value_counts.values,
                                    color_continuous_scale='Plasma',
                                    template='plotly_white')
                    else:
                        fig = px.bar(df, x=selected_col, title=f"Bar Chart de {selected_col}",
                                    color_discrete_sequence=['#FFD93D'],
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
                                title=f"Pie Chart de {selected_col}",
                                color_discrete_sequence=px.colors.qualitative.Set3,
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
        
        ch = st.columns(3)  # Ajout d'une colonne pour le choix du type de graphique
        x_col = ch[0].selectbox(label='X_colonne', options=df.columns)
        y_col = ch[1].selectbox(label='Y_colonne', options=df.columns)

        # Vérifier les types de données pour proposer des graphiques appropriés
        if x_col and y_col and not df.empty:
            x_is_object = df[x_col].dtype == 'object' or df[x_col].dtype.name == 'category' or df[x_col].nunique() <= 10
            y_is_object = df[y_col].dtype == 'object' or df[y_col].dtype.name == 'category' or df[y_col].nunique() <= 10

            # Proposer différents types de graphiques selon les types de données
            if x_is_object and y_is_object:
                # Les deux sont catégorielles - proposer des graphiques adaptés
                graph_type = ch[2].selectbox(
                    label='Type de graphique',
                    options=['Bar Chart', 'Count Plot', 'Heatmap'],
                    key='cat_cat'
                )

                if graph_type == 'Bar Chart':
                    fig = px.bar(df, x=x_col, color=y_col, title=f"Bar Chart of {x_col} by {y_col}",
                                color_discrete_sequence=px.colors.qualitative.Pastel,
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
                    # Créer un tableau croisé pour le countplot
                    cross_tab = pd.crosstab(df[x_col], df[y_col])
                    fig = px.imshow(cross_tab, text_auto=True, aspect="auto",
                                    title=f"Count Plot of {x_col} vs {y_col}",
                                    color_continuous_scale='Blues',
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
                    # Normaliser le tableau croisé pour avoir des pourcentages
                    cross_tab_norm = pd.crosstab(df[x_col], df[y_col], normalize='index')
                    fig = px.imshow(cross_tab_norm, text_auto='.1%', aspect="auto",
                                    color_continuous_scale='RdYlBu_r',
                                    title=f"Heatmap of {x_col} vs {y_col} (% par ligne)",
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

                graph_type = ch[2].selectbox(
                    label='Type de graphique',
                    options=['Box Plot', 'Violin Plot', 'Bar Chart', 'Swarm Plot'],
                    key='cat_num'
                )

                if graph_type == 'Box Plot':
                    fig = px.box(df, x=cat_col, y=num_col,
                                title=f"Box Plot of {num_col} by {cat_col}",
                                color=cat_col,
                                color_discrete_sequence=px.colors.qualitative.Pastel,
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
                                    title=f"Violin Plot of {num_col} by {cat_col}",
                                    color=cat_col,
                                    color_discrete_sequence=px.colors.qualitative.Set2,
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
                    # Calculer la moyenne par catégorie
                    agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
                    fig = px.bar(agg_df, x=cat_col, y=num_col,
                                title=f"Average {num_col} by {cat_col}",
                                color=num_col,
                                color_continuous_scale='Cividis',
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
                    # Pour un swarm plot, utiliser Plotly Strip Plot comme alternative
                    fig = px.strip(df, x=cat_col, y=num_col,
                                title=f"Swarm Plot of {num_col} by {cat_col}",
                                color=cat_col,
                                color_discrete_sequence=px.colors.qualitative.Dark2,
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
                graph_type = ch[2].selectbox(
                    label='Type de graphique',
                    options=['Scatter Plot', 'Line Plot', 'Hexbin', 'Density Contour', 'Bubble Chart'],
                    key='num_num'
                )

                if graph_type == 'Scatter Plot':
                    fig = px.scatter(df, x=x_col, y=y_col,
                                    title=f"Scatter Plot of {y_col} vs {x_col}",
                                    color_discrete_sequence=['#FF6B6B'],
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
                    # Trier par x pour avoir une ligne cohérente
                    sorted_df = df.sort_values(by=x_col)
                    fig = px.line(sorted_df, x=x_col, y=y_col,
                                title=f"Line Plot of {y_col} vs {x_col}",
                                template='plotly_white')
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family="Arial, sans-serif", size=12),
                        title_font_size=16,
                        title_x=0.5
                    )
                    fig.update_traces(
                        line=dict(color='#4ECDC4', width=3),
                        marker=dict(size=6, color='#FF6B6B', line=dict(width=1, color='white'))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif graph_type == 'Hexbin':
                    fig = px.density_heatmap(df, x=x_col, y=y_col, nbinsx=20, nbinsy=20,
                                            title=f"Hexbin Plot of {y_col} vs {x_col}",
                                            color_continuous_scale='Hot',
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
                                            title=f"Density Contour of {y_col} vs {x_col}")
                    st.plotly_chart(fig)
                elif graph_type == 'Bubble Chart':
                    # Utiliser une troisième variable numérique pour la taille des bulles si disponible
                    num_cols = df.select_dtypes(include=['float', 'int']).columns
                    if len(num_cols) > 2 and set([x_col, y_col]).issubset(set(num_cols)):
                        size_col = [col for col in num_cols if col not in [x_col, y_col]][0]
                        fig = px.scatter(df, x=x_col, y=y_col, size=size_col,
                                        title=f"Bubble Chart of {y_col} vs {x_col} (size: {size_col})")
                    else:
                        # Si pas de 3ème variable numérique, utiliser une constante
                        fig = px.scatter(df, x=x_col, y=y_col, size_max=15,
                                        title=f"Bubble Chart of {y_col} vs {x_col}")
                    st.plotly_chart(fig)

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
