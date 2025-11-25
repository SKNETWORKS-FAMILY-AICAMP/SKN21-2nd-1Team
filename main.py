import streamlit as st

pages = {
    "최종 모델 리포트": [
        st.Page("ne_eun/view.py", title="최종모델", url_path="final_model"),
    ],
    "비교 모델 리포트": [
        st.Page("ne_eun/view.py", title="박내은", url_path="neeun"),
        st.Page("grkim/app.py", title="김가람", url_path="garam"),
        st.Page("yiseon/Streamlit.py", title="장이선", url_path="yiseon"),
    ],
}
pg = st.navigation(pages)
pg.run()
