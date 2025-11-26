import streamlit as st

pages = {
    "최종 모델 리포트": [
        st.Page("ne_eun/view.py", title="최종모델", url_path="final_model"),
    ],
    "비교 모델 리포트": [
        st.Page("ne_eun/view.py", title="박내은", url_path="neeun"),
        st.Page("grkim/app.py", title="김가람", url_path="garam"),
        st.Page("yiseon/yiseon.py", title="장이선", url_path="yiseon"),
        st.Page("dkjung/streamlit/mainboard.py", title="정덕규", url_path="dkjung"),
        st.Page("hyeun_uk/streamlit.py", title="강현욱", url_path="hyeun_uk"),
    ],
}
pg = st.navigation(pages)
pg.run()
