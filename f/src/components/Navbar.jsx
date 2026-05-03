import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav className="navbar">
      <div className="logo">Sign Language Translator</div>

      <div className="nav-links">
        <Link to="/">Home</Link>
        <a href="#languages">Languages</a>
      </div>
    </nav>
  );
}

export default Navbar;