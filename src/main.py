# # Remove all relative imports (dots)
# from data_models import DataSchema, DataGenerator
# from recommendation import MentorRecommender
# from model_manager import ModelManager
# from evaluation import RecommendationEvaluator

# import pandas as pd
# import streamlit as st

# # -----------------------------
# # Streamlit Application
# # -----------------------------
# def main():
#     st.set_page_config(page_title="CLAT Mentor Recommender", layout="wide")
    
#     # Load/generate data
#     aspirants_df = DataGenerator.generate_aspirants(200)
#     mentors_df = DataGenerator.generate_mentors(50)
#     interactions_df = DataGenerator.generate_interaction_data(aspirants_df, mentors_df)
    
#     # Initialize recommender
#     recommender = MentorRecommender(mentors_df, interactions_df)
#     recommender.preprocess_features()
    
#     st.title("CLAT Mentor Recommendation System")
#     st.markdown("""
#     ### Find the perfect mentor for your CLAT preparation
#     Provide your details below to get personalized mentor recommendations.
#     """)
    
#     with st.form("aspirant_profile"):
#         col1, col2 = st.columns(2)
        
#         with col1:
#             name = st.text_input("Name")
#             prep_level = st.selectbox("Preparation Level", DataSchema.PREPARATION_LEVELS)
#             learning_style = st.selectbox("Learning Style", DataSchema.LEARNING_STYLES)
#             budget = st.checkbox("Budget Constraints")
            
#         with col2:
#             subjects = st.multiselect("Preferred Subjects", DataSchema.SUBJECTS)
#             colleges = st.multiselect("Target Colleges", DataSchema.COLLEGES)
#             weaknesses = st.multiselect("Your Weaknesses", DataSchema.WEAKNESSES)
#             languages = st.multiselect("Preferred Languages", DataSchema.LANGUAGES)
            
#         submitted = st.form_submit_button("Get Recommendations")
    
#     if submitted:
#         aspirant_data = {
#             "name": name,
#             "preferred_subjects": subjects,
#             "target_colleges": colleges,
#             "preparation_level": prep_level,
#             "learning_style": learning_style,
#             "weaknesses": weaknesses,
#             "languages": languages,
#             "budget_constraint": budget
#         }
        
#         # Get recommendations
#         recommendations, explanations = recommender.hybrid_recommendation(aspirant_data, top_k=5)
        
#         # Display results
#         st.subheader("Recommended Mentors")
#         for idx, (_, mentor) in enumerate(recommendations.iterrows()):
#             with st.expander(f"{mentor['name']} ⭐ {mentor['rating']} | ₹{mentor['hourly_rate']}/hr"):
#                 col_left, col_right = st.columns([1, 3])
                
#                 with col_left:
#                     st.markdown(f"**Expertise Level:** {mentor['expertise_level']}")
#                     st.markdown(f"**Experience:** {mentor['years_experience']} years")
#                     st.markdown(f"**Availability:** {mentor['availability_hours_per_week']} hrs/week")
                
#                 with col_right:
#                     expl = explanations.get(mentor['mentor_id'], {})
#                     reasons = []
#                     if expl.get('subject_matches'):
#                         reasons.append(f"📘 **Subjects:** {', '.join(expl['subject_matches'])}")
#                     if expl.get('college_matches'):
#                         reasons.append(f"🏛 **Colleges:** {', '.join(expl['college_matches'])}")
#                     if expl.get('style_match'):
#                         reasons.append(f"🎯 **Learning Style Match**")
#                     if expl.get('weakness_coverage'):
#                         reasons.append(f"🔧 **Weakness Help:** {', '.join(expl['weakness_coverage'])}")
                    
#                     st.markdown("\n".join(reasons))
                
#                 st.markdown(f"**Strengths:** {', '.join(mentor['strengths'])}")
#                 st.markdown(f"**Bio:** {mentor['bio']}")


# if __name__ == "__main__":
#     main()



from data_models import DataSchema, DataGenerator
from recommendation import MentorRecommender
import pandas as pd
import streamlit as st

def main():
    st.set_page_config(page_title="CLAT Mentor Recommender", layout="wide")
    
    # Initialize data
    aspirants_df = DataGenerator.generate_aspirants(200)
    mentors_df = DataGenerator.generate_mentors(50)
    interactions_df = DataGenerator.generate_interaction_data(aspirants_df, mentors_df)
    
    # Initialize recommender
    recommender = MentorRecommender(mentors_df, interactions_df)
    recommender.preprocess_features()
    
    # UI Components
    st.title("CLAT Mentor Recommendation System")
    st.markdown("### Find the perfect mentor for your CLAT preparation")
    
    with st.form("aspirant_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Your Name")
            prep_level = st.selectbox("Current Preparation Level", DataSchema.PREPARATION_LEVELS)
            learning_style = st.selectbox("Preferred Learning Style", DataSchema.LEARNING_STYLES)
            budget = st.checkbox("Budget Constraints (₹1000/hr or less)")
            
        with col2:
            subjects = st.multiselect("Focus Subjects", DataSchema.SUBJECTS)
            colleges = st.multiselect("Target NLUs", DataSchema.COLLEGES)
            weaknesses = st.multiselect("Areas Needing Improvement", DataSchema.WEAKNESSES)
            languages = st.multiselect("Preferred Languages", DataSchema.LANGUAGES)
            
        submitted = st.form_submit_button("Find My Mentors")
    
    if submitted:
        aspirant_data = {
            "name": name,
            "preferred_subjects": subjects,
            "target_colleges": colleges,
            "preparation_level": prep_level,
            "learning_style": learning_style,
            "weaknesses": weaknesses,
            "languages": languages,
            "budget_constraint": budget
        }
        
        # Get recommendations
        recommendations, explanations = recommender.hybrid_recommendation(
            aspirant_data, 
            top_k=5
        )
        
        # Display results
        st.subheader("Top Mentor Recommendations")
        for _, mentor in recommendations.iterrows():
            mentor_id = mentor['mentor_id']
            with st.expander(f"{mentor['name']} ⭐ {mentor['rating']}/5 | ₹{mentor['hourly_rate']}/hour"):
                cols = st.columns([1, 3])
                
                with cols[0]:
                    st.metric("Experience", f"{mentor['years_experience']} years")
                    st.metric("Availability", f"{mentor['availability_hours_per_week']} hrs/week")
                    st.metric("CLAT Score", mentor['clat_score'])
                
                with cols[1]:
                    expl = explanations.get(mentor_id, {})
                    reasons = []
                    
                    if expl.get('subject_matches'):
                        reasons.append(f"📚 **Matching Subjects:** {', '.join(expl['subject_matches'])}")
                    if expl.get('college_matches'):
                        reasons.append(f"🏛 **Alma Maters:** {', '.join(expl['college_matches'])}")
                    if expl.get('style_match'):
                        reasons.append("🎯 **Learning Style Match**")
                    if expl.get('weakness_coverage'):
                        reasons.append(f"🔧 **Weakness Coverage:** {', '.join(expl['weakness_coverage'])}")
                    
                    st.markdown("\n".join(reasons))
                    st.markdown(f"**Strengths:** {', '.join(mentor['strengths'])}")
                    st.markdown(f"*{mentor['bio']}*")

if __name__ == "__main__":
    main()