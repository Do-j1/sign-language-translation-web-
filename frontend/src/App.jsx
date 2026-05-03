import { Routes, Route } from "react-router-dom";
import LanguageSelect from "./components/LanguageSelect";
import Translator from "./components/Translator";

function App() {
  return (
    <Routes>
      <Route path="/" element={<LanguageSelect />} />
      <Route path="/translator/:lang" element={<Translator />} />
    </Routes>
  );
}

export default App;