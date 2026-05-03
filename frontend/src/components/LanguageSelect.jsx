import { useNavigate } from "react-router-dom";
import Navbar from "./Navbar";
import Footer from "./Footer";

function LanguageSelect() {
  const navigate = useNavigate();

  const handleSelectLanguage = (lang) => {
    navigate(`/translator/${lang}`);
  };

  return (
    <div className="page">
      <Navbar />

      <main className="hero" id="languages">
        <h1>Choose Your Sign Language</h1>
        <p>
          Select the sign language first so the system can load the correct
          dataset or model, then continue to the live translation page.
        </p>

        <div className="language-grid">
          <div
            className="language-card"
            onClick={() => handleSelectLanguage("english")}
          >
            <h3>English Alphabet</h3>
            <p>
              Detect hand gestures for English letters and alphabet-based signs.
            </p>
            <button className="btn btn-primary">Choose English</button>
          </div>

          <div
            className="language-card"
            onClick={() => handleSelectLanguage("arabic")}
          >
            <h3>Arabic Sign Language</h3>
            <p>
              Load Arabic sign language data and model for Arabic translation.
            </p>
            <button className="btn btn-primary">Choose Arabic</button>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default LanguageSelect;