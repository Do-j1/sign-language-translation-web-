import { Link, useParams } from "react-router-dom";
import { useEffect, useRef, useState } from "react";
import Navbar from "./Navbar";
import Footer from "./Footer";

function Translator() {
	const { lang } = useParams();

	const videoRef = useRef(null);
	const canvasRef = useRef(null);
	const streamRef = useRef(null);
	const handsRef = useRef(null);
	const cameraUtilsRef = useRef(null);
	const lastSentTimeRef = useRef(0);

	const noHandSinceRef = useRef(null);
	const spaceAddedRef = useRef(false);

	// English direct mode
	const englishFrameHistoryRef = useRef([]);
	const englishAppendedLetterRef = useRef(null);
	const englishStableSinceRef = useRef(null);

	// Arabic words direct mode
	const sequenceFramesRef = useRef([]);
	const lastAppendedWordRef = useRef("");
	const lastWordAppendTimeRef = useRef(0);

	// Arabic hybrid mode
	const arabicSegmentLettersRef = useRef([]);
	const arabicLetterHistoryRef = useRef([]);
	const arabicStableSinceRef = useRef(null);
	const arabicLastAcceptedLetterRef = useRef(null);
	const arabicWordVotesRef = useRef({});
	const arabicWordStableSinceRef = useRef(null);
	const arabicLastBestWordRef = useRef("");
	const arabicSegmentStartedRef = useRef(false);
	const arabicSegmentModeRef = useRef(null); // null | "letters" | "word"

	// Holds the last confident prediction so noisy frames don't flash "---"
	const lastGoodPredictionRef = useRef("---");

	const [cameraOn, setCameraOn] = useState(false);
	const [error, setError] = useState("");
	const [handDetected, setHandDetected] = useState(false);
	const [translatedText, setTranslatedText] = useState("---");
	const [confidence, setConfidence] = useState(null);
	const [top3, setTop3] = useState([]);
	const [builtText, setBuiltText] = useState("");
	const [currentSegmentPreview, setCurrentSegmentPreview] = useState("");
	const [flashCapture, setFlashCapture] = useState(false);
	const [isUnsure, setIsUnsure] = useState(false);

	const [manualArabicMode, setManualArabicMode] = useState("auto"); // "auto" | "letters" | "words"
	const manualArabicModeRef = useRef("auto");

	const supportedLanguages = ["arabic", "english", "arabic_words"];

	const ARABIC_WORD_MARGIN_NO_LETTERS = 0.28;
	const ARABIC_WORD_MARGIN_ONE_LETTER = 0.35;
	const ARABIC_WORD_VOTE_THRESHOLD = 8;
	const ARABIC_WORD_STABLE_TIME_MS = 400;
	const isValidLanguage = supportedLanguages.includes(lang);

	const API_URL = import.meta.env.VITE_API_URL;
	const REQUEST_INTERVAL_MS = 400; // faster polling

	// ── Letter speed knobs (faster) ──────────────────────────────────────────
	const NO_HAND_DELAY_MS = 700;
	const LETTER_REQUIRED_FRAMES = 4;
	const LETTER_MIN_CONFIDENCE = 0.60;
	const LETTER_TOP3_MARGIN = 0.18;
	const LETTER_STABLE_TIME_MS = 150;
	const LETTER_UNSURE_TIME_MS = 250;
	// ─────────────────────────────────────────────────────────────────────────

	// ── Word sequence speed knobs ─────────────────────────────────────────────
	// FIX 2: Lowered sequence length (5→3) and confidence (0.72→0.65) so
	//        arabic_words direct mode commits predictions much faster.
	const WORDS_SEQUENCE_LENGTH = 3;
	const WORDS_MIN_CONFIDENCE = 0.50;
	const WORDS_APPEND_COOLDOWN_MS = 2000; // meaningful cooldown to avoid double-appending same word
	// ─────────────────────────────────────────────────────────────────────────

	const ARABIC_WORD_MIN_VOTES = 3;
	const ARABIC_WORD_MIN_AVG_CONF = 0.7;
	const ARABIC_WORD_MIN_MAX_CONF = 0.88;
	const ARABIC_WORD_EARLY_LOCK_CONF = 0.92;
	const ARABIC_WORD_EARLY_LOCK_VOTES = 3;

	// How long hand must be absent before we finalize the arabic segment
	const ARABIC_NO_HAND_FINALIZE_MS = 650;

	const getLanguageName = () => {
		if (lang === "arabic") return "Arabic Sign Language";
		if (lang === "english") return "English Sign Language";
		if (lang === "arabic_words") return "Arabic Sign Words";
		return "Unknown Language";
	};

	const getDatasetName = () => {
		if (lang === "arabic") return "Arabic Hybrid Model";
		if (lang === "english") return "English Letters Model";
		if (lang === "arabic_words") return "Arabic Words Model";
		return "Unknown Model";
	};

	const loadMediaPipe = async () => {
		const loadScript = (src) =>
			new Promise((resolve, reject) => {
				if (document.querySelector(`script[src="${src}"]`)) {
					resolve();
					return;
				}
				const script = document.createElement("script");
				script.src = src;
				script.onload = resolve;
				script.onerror = () => reject(new Error(`Failed to load: ${src}`));
				document.body.appendChild(script);
			});

		await loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js");
		await loadScript(
			"https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js"
		);
		await loadScript(
			"https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js"
		);
	};

	const preprocessLandmarks = (points) => {
		if (!points || !Array.isArray(points) || points.length === 0) return [];
		return points.flatMap((point) => [
			Number(point.x.toFixed(6)),
			Number(point.y.toFixed(6)),
			Number(point.z.toFixed(6)),
		]);
	};

	const triggerFlash = () => {
		setFlashCapture(true);
		setTimeout(() => setFlashCapture(false), 300);
	};

	const resetPredictionState = () => {
		lastGoodPredictionRef.current = "---";
		setHandDetected(false);
		setTranslatedText("---");
		setConfidence(null);
		setTop3([]);
		setIsUnsure(false);
	};

	const resetEnglishTracking = () => {
		englishFrameHistoryRef.current = [];
		englishAppendedLetterRef.current = null;
		englishStableSinceRef.current = null;
		setIsUnsure(false);
	};

	const resetArabicWordsTracking = () => {
		sequenceFramesRef.current = [];
		lastAppendedWordRef.current = "";
		lastWordAppendTimeRef.current = 0;
	};

	const resetArabicHybridTracking = () => {
		arabicSegmentLettersRef.current = [];
		arabicLetterHistoryRef.current = [];
		arabicStableSinceRef.current = null;
		arabicLastAcceptedLetterRef.current = null;
		arabicWordVotesRef.current = {};
		arabicWordStableSinceRef.current = null;
		arabicLastBestWordRef.current = "";
		arabicSegmentStartedRef.current = false;
		arabicSegmentModeRef.current = null;
		sequenceFramesRef.current = [];
		setCurrentSegmentPreview("");
		setIsUnsure(false);
	};

	const clearBuiltText = () => {
		builtTextRef.current = "";
		setBuiltText("");
		setCurrentSegmentPreview("");
		resetPredictionState();
		resetEnglishTracking();
		resetArabicWordsTracking();
		resetArabicHybridTracking();
		noHandSinceRef.current = null;
		spaceAddedRef.current = false;
	};

	const deleteLastCharacter = () => {
		setBuiltText((prev) => {
			const next = prev.slice(0, -1);
			builtTextRef.current = next;
			return next;
		});
	};

	const addSpace = () => {
		if (lang === "arabic") {
			finalizeArabicSegment();
			return;
		}
		setBuiltTextSynced((prev) => {
			if (!prev || prev.endsWith(" ")) return prev;
			return `${prev} `;
		});
		if (lang === "english") resetEnglishTracking();
		if (lang === "arabic_words") resetArabicWordsTracking();
		noHandSinceRef.current = null;
		spaceAddedRef.current = false;
	};

	const pushFrameToSequence = (points) => {
		const processedPoints = preprocessLandmarks(points);
		if (processedPoints.length !== 63) return null;
		sequenceFramesRef.current.push(processedPoints);
		if (sequenceFramesRef.current.length > WORDS_SEQUENCE_LENGTH) {
			sequenceFramesRef.current.shift();
		}
		return processedPoints;
	};

	const fetchPrediction = async (payload) => {
		const response = await fetch(`${API_URL}/predict`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify(payload),
		});
		const data = await response.json();
		if (!response.ok) throw new Error(data.error || "Prediction failed.");
		return data;
	};

	const getBestArabicWordStats = () => {
		const wordEntries = Object.entries(arabicWordVotesRef.current || {});
		if (wordEntries.length === 0) return null;
		return wordEntries
			.map(([word, stats]) => ({ word, ...stats }))
			.sort((a, b) => {
				if (b.count !== a.count) return b.count - a.count;
				if (b.avgConf !== a.avgConf) return b.avgConf - a.avgConf;
				return b.maxConf - a.maxConf;
			})[0];
	};



	const isVeryStrongArabicWordCandidate = (bestWord) =>
		Boolean(
			bestWord &&
			bestWord.count >= ARABIC_WORD_VOTE_THRESHOLD + 2 &&
			bestWord.avgConf >= 0.94 &&
			bestWord.maxConf >= 0.96
		);

	const shouldUseWordPrediction = (
		lettersPreview,
		wordConfidence,
		letterConfidence,
		bestWord
	) => {
		if (
			typeof wordConfidence !== "number" ||
			typeof letterConfidence !== "number"
		) return false;

		const wordStableDuration = arabicWordStableSinceRef.current
			? Date.now() - arabicWordStableSinceRef.current
			: 0;
		const stableEnough = wordStableDuration >= ARABIC_WORD_STABLE_TIME_MS;

		if (!lettersPreview) {
			return (
				stableEnough &&
				bestWord &&
				bestWord.count >= ARABIC_WORD_VOTE_THRESHOLD - 2 &&
				bestWord.avgConf >= 0.88 &&
				bestWord.maxConf >= 0.9 &&
				wordConfidence > letterConfidence + ARABIC_WORD_MARGIN_NO_LETTERS &&
				wordConfidence >= 0.88
			);
		}

		if (lettersPreview.length >= 2) return false;
		if (!bestWord) return false;

		return (
			stableEnough &&
			bestWord.count >= ARABIC_WORD_VOTE_THRESHOLD &&
			bestWord.avgConf >= 0.92 &&
			bestWord.maxConf >= 0.94 &&
			wordConfidence > letterConfidence + ARABIC_WORD_MARGIN_ONE_LETTER &&
			wordConfidence >= 0.95
		);
	};

	const buildArabicPreview = () => {
		const lettersPreview = arabicSegmentLettersRef.current.join("");
		const bestWord = getBestArabicWordStats();
		const currentMode =
			manualArabicModeRef.current === "auto"
				? arabicSegmentModeRef.current
				: manualArabicModeRef.current;

		if (currentMode === "letters") return lettersPreview;

		if (currentMode === "words") {
			const wordStableDuration = arabicWordStableSinceRef.current
				? Date.now() - arabicWordStableSinceRef.current
				: 0;
			if (wordStableDuration >= ARABIC_WORD_STABLE_TIME_MS && bestWord) {
				return bestWord.word;
			}
			return "---";
		}

		if (lettersPreview) return lettersPreview;

		if (bestWord && bestWord.count >= 2 && bestWord.avgConf >= 0.8) {
			return bestWord.word;
		}

		return "";
	};

	const updateArabicPreviewState = () => {
		setCurrentSegmentPreview(buildArabicPreview());
	};

	const shouldPreferArabicWordOverSingleLetter = (lettersText, bestWord) => {
		if (!lettersText || lettersText.length !== 1 || !bestWord) return false;
		const strongWord =
			bestWord.count >= ARABIC_WORD_VOTE_THRESHOLD - 2 &&
			bestWord.avgConf >= 0.88 &&
			bestWord.maxConf >= 0.9;
		const fallbackWord =
			bestWord.count >= 3 &&
			bestWord.avgConf >= 0.84 &&
			bestWord.maxConf >= 0.88;
		return strongWord || fallbackWord;
	};

	const toggleArabicMode = () => {
		const newMode =
			manualArabicMode === "auto"
				? "letters"
				: manualArabicMode === "letters"
					? "words"
					: "auto";
		setManualArabicMode(newMode);
		manualArabicModeRef.current = newMode;
		resetArabicHybridTracking();
	};

	const getDisplayedBuiltText = () => {
		if (lang === "arabic") {
			const preview = currentSegmentPreview || "";
			const committed = builtText || "";
			if (!preview) return committed || "---";
			if (!committed) return preview;
			return `${committed}${preview}`;
		}
		return builtText || "---";
	};

	const maybeLockArabicWordMode = () => {
		if (arabicSegmentModeRef.current === "letters") return;

		const bestWord = getBestArabicWordStats();
		const hasLetters = arabicSegmentLettersRef.current.length > 1;
		if (hasLetters) return;

		const wordStableDuration = arabicWordStableSinceRef.current
			? Date.now() - arabicWordStableSinceRef.current
			: 0;

		if (
			isVeryStrongArabicWordCandidate(bestWord) &&
			wordStableDuration >= ARABIC_WORD_STABLE_TIME_MS
		) {
			arabicSegmentModeRef.current = "word";
			updateArabicPreviewState();
		}
	};

	const tryAutoFinalizeArabicSegment = (bestWord) => {
		if (!bestWord || !bestWord.word) return;

		const mode =
			manualArabicModeRef.current === "auto"
				? arabicSegmentModeRef.current
				: manualArabicModeRef.current;
		if (mode === "letters") return;

		const lettersText = arabicSegmentLettersRef.current.join("").trim();
		if (lettersText.length >= 1) return;

		const wordStableDuration = arabicWordStableSinceRef.current
			? Date.now() - arabicWordStableSinceRef.current
			: 0;

		const canCommitWord =
			bestWord.count >= ARABIC_WORD_VOTE_THRESHOLD - 2 &&
			bestWord.avgConf >= 0.88 &&
			bestWord.maxConf >= 0.9 &&
			wordStableDuration >= 400;

		if (mode === "words" && canCommitWord) {
			finalizeArabicSegment();
			return;
		}

		const oneLetterPreview = lettersText.length === 1;
		if (mode === "auto" && oneLetterPreview && canCommitWord) {
			finalizeArabicSegment();
		}
	};

	const collectArabicLetterEvidence = (letter, conf, top3Data) => {
		if (manualArabicModeRef.current === "words") return;
		if (!letter || letter === "---" || letter === "غير واضح") return;
		if (typeof conf !== "number" || conf < LETTER_MIN_CONFIDENCE) return;

		arabicSegmentStartedRef.current = true;

		const unsure =
			top3Data &&
			top3Data.length >= 2 &&
			top3Data[0].score - top3Data[1].score < LETTER_TOP3_MARGIN;

		setIsUnsure(unsure);

		const requiredTime = unsure ? LETTER_UNSURE_TIME_MS : LETTER_STABLE_TIME_MS;

		arabicLetterHistoryRef.current.push(letter);
		if (arabicLetterHistoryRef.current.length > LETTER_REQUIRED_FRAMES) {
			arabicLetterHistoryRef.current.shift();
		}

		const allSame =
			arabicLetterHistoryRef.current.length === LETTER_REQUIRED_FRAMES &&
			arabicLetterHistoryRef.current.every((item) => item === letter);

		if (!allSame) return;

		if (
			arabicLastAcceptedLetterRef.current !== letter &&
			arabicStableSinceRef.current === null
		) {
			arabicStableSinceRef.current = Date.now();
		}

		const stableDuration =
			Date.now() - (arabicStableSinceRef.current || Date.now());

		if (stableDuration < requiredTime) return;
		if (arabicLastAcceptedLetterRef.current === letter) return;

		arabicSegmentLettersRef.current.push(letter);
		arabicLastAcceptedLetterRef.current = letter;
		arabicStableSinceRef.current = null;
		arabicLetterHistoryRef.current = [];

		const lettersText = arabicSegmentLettersRef.current.join("").trim();
		const bestWord = getBestArabicWordStats();

		if (arabicSegmentLettersRef.current.length >= 2) {
			arabicSegmentModeRef.current = "letters";
		} else if (arabicSegmentModeRef.current === null) {
			if (!shouldPreferArabicWordOverSingleLetter(lettersText, bestWord)) {
				arabicSegmentModeRef.current = "letters";
			}
		}

		updateArabicPreviewState();
	};

	const collectArabicWordEvidence = (word, conf) => {
		if (manualArabicModeRef.current === "letters") return;
		if (arabicSegmentLettersRef.current.length > 0) return;
		if (!word || word === "---") return;
		if (typeof conf !== "number" || conf < 0.7) return;

		arabicSegmentStartedRef.current = true;

		const votes = arabicWordVotesRef.current;
		if (!votes[word]) {
			votes[word] = { count: 0, sumConf: 0, avgConf: 0, maxConf: 0 };
		}
		votes[word].count += 1;
		votes[word].sumConf += conf;
		votes[word].avgConf = votes[word].sumConf / votes[word].count;
		votes[word].maxConf = Math.max(votes[word].maxConf, conf);
		arabicWordVotesRef.current = votes;

		const currentBestWord = getBestArabicWordStats();
		if (currentBestWord?.word === arabicLastBestWordRef.current) {
			if (!arabicWordStableSinceRef.current) {
				arabicWordStableSinceRef.current = Date.now();
			}
		} else {
			arabicWordStableSinceRef.current = Date.now();
			arabicLastBestWordRef.current = currentBestWord?.word || "";
		}

		maybeLockArabicWordMode();
		updateArabicPreviewState();
		tryAutoFinalizeArabicSegment(currentBestWord);
	};

	const chooseArabicSegmentResult = () => {
		const lettersText = arabicSegmentLettersRef.current.join("").trim();
		const bestWord = getBestArabicWordStats();
		const currentMode =
			manualArabicModeRef.current === "auto"
				? arabicSegmentModeRef.current
				: manualArabicModeRef.current;

		if (currentMode === "letters") return lettersText;

		if (currentMode === "words") {
			// FIX 2a: Relaxed threshold — isStrongArabicWordCandidate required avgConf >= 0.92
			// which was rarely met. Match the no-hand finalize bar instead.
			if (bestWord && bestWord.count >= 3 && bestWord.avgConf >= 0.65) {
				return bestWord.word;
			}
			return "";
		}

		// auto mode
		if (!bestWord) return lettersText;
		if (lettersText) return lettersText;
		if (bestWord.count >= 5 && bestWord.avgConf >= 0.82) return bestWord.word;
		return "";
	};

	const finalizeArabicSegment = () => {
		const finalSegment = chooseArabicSegmentResult();

		if (!finalSegment) {
			resetArabicHybridTracking();
			noHandSinceRef.current = null;
			spaceAddedRef.current = false;
			return;
		}

		// FIX 2b: Use setBuiltTextSynced so builtTextRef stays in sync.
		setBuiltTextSynced((prev) => {
			const trimmed = prev.trimEnd();
			return trimmed ? `${trimmed} ${finalSegment} ` : `${finalSegment} `;
		});

		triggerFlash();
		resetArabicHybridTracking();
		noHandSinceRef.current = null;
		spaceAddedRef.current = false;
	};

	const tryAutoAppendEnglishLetter = (letter, conf, top3Data) => {
		if (!letter || letter === "---") return;
		if (typeof conf !== "number" || conf < LETTER_MIN_CONFIDENCE) return;

		const unsure =
			top3Data &&
			top3Data.length >= 2 &&
			top3Data[0].score - top3Data[1].score < LETTER_TOP3_MARGIN;

		setIsUnsure(unsure);

		const requiredTime = unsure ? LETTER_UNSURE_TIME_MS : LETTER_STABLE_TIME_MS;

		englishFrameHistoryRef.current.push(letter);
		if (englishFrameHistoryRef.current.length > LETTER_REQUIRED_FRAMES) {
			englishFrameHistoryRef.current.shift();
		}

		const allSame =
			englishFrameHistoryRef.current.length === LETTER_REQUIRED_FRAMES &&
			englishFrameHistoryRef.current.every((item) => item === letter);

		if (!allSame) return;

		if (
			englishAppendedLetterRef.current !== letter &&
			englishStableSinceRef.current === null
		) {
			englishStableSinceRef.current = Date.now();
		}

		const stableDuration =
			Date.now() - (englishStableSinceRef.current || Date.now());

		if (stableDuration < requiredTime) return;
		if (englishAppendedLetterRef.current === letter) return;

		setBuiltTextSynced((prev) => prev + letter);
		englishAppendedLetterRef.current = letter;
		englishStableSinceRef.current = null;
		englishFrameHistoryRef.current = [];

		triggerFlash();
	};

	// Ref that mirrors builtText so dedup checks are always synchronous
	const builtTextRef = useRef("");

	const setBuiltTextSynced = (updaterOrValue) => {
		setBuiltText((prev) => {
			const next =
				typeof updaterOrValue === "function" ? updaterOrValue(prev) : updaterOrValue;
			builtTextRef.current = next;
			return next;
		});
	};

	// ── FIX 1: Dedup is synchronous via builtTextRef so async setState batching
	//    never causes phantom duplicates. The same-word lock resets when a
	//    different word arrives, so "marhaba → ana → marhaba" works correctly.
	const tryAutoAppendArabicWordDirect = (word, conf) => {
		if (!word || word === "---") return;
		if (typeof conf !== "number" || conf < WORDS_MIN_CONFIDENCE) return;

		const now = Date.now();

		if (lastAppendedWordRef.current !== word) {
			lastAppendedWordRef.current = "";
		}

		if (
			lastAppendedWordRef.current === word &&
			now - lastWordAppendTimeRef.current < WORDS_APPEND_COOLDOWN_MS
		) {
			return;
		}

		const trimmed = builtTextRef.current.trim();
		const tokens = trimmed.split(/\s+/).filter(Boolean);
		const lastToken = tokens[tokens.length - 1] || "";
		if (lastToken === word) {
			lastAppendedWordRef.current = word;
			lastWordAppendTimeRef.current = now;
			return;
		}

		const next = trimmed ? `${trimmed} ${word}` : word;
		builtTextRef.current = next;
		setBuiltText(next);

		lastAppendedWordRef.current = word;
		lastWordAppendTimeRef.current = now;

		triggerFlash();
	};

	const handleArabicHybrid = async (processedPoints) => {
		let letterData = null;
		let wordData = null;

		if (manualArabicModeRef.current !== "words") {
			letterData = await fetchPrediction({
				language: "arabic",
				landmarks: processedPoints,
			});
		}

		if (
			manualArabicModeRef.current !== "letters" &&
			sequenceFramesRef.current.length >= 1
		) {
			wordData = await fetchPrediction({
				language: "arabic_words",
				sequence_landmarks: sequenceFramesRef.current,
			});
		}

		const letterPrediction = letterData?.prediction || "---";
		const letterConfidence =
			typeof letterData?.confidence === "number" ? letterData.confidence : null;
		const letterTop3 = Array.isArray(letterData?.top3) ? letterData.top3 : [];

		const wordPrediction = wordData?.prediction || "---";
		const wordConfidence =
			typeof wordData?.confidence === "number" ? wordData.confidence : null;

		collectArabicLetterEvidence(letterPrediction, letterConfidence, letterTop3);

		if (wordData) {
			collectArabicWordEvidence(wordPrediction, wordConfidence);
		}

		let livePrediction = letterPrediction;
		let liveConfidence = letterConfidence;
		let liveTop3 = letterTop3;
		const lettersPreview = arabicSegmentLettersRef.current.join("").trim();
		const bestWord = getBestArabicWordStats();
		const currentMode =
			manualArabicModeRef.current === "auto"
				? arabicSegmentModeRef.current
				: manualArabicModeRef.current;

		if (currentMode === "words" && wordData) {
			livePrediction = wordPrediction;
			liveConfidence = wordConfidence;
			liveTop3 = Array.isArray(wordData?.top3) ? wordData.top3 : [];
		} else if (
			currentMode === "auto" &&
			wordData &&
			shouldUseWordPrediction(lettersPreview, wordConfidence, letterConfidence, bestWord)
		) {
			livePrediction = wordPrediction;
			liveConfidence = wordConfidence;
			liveTop3 = Array.isArray(wordData?.top3) ? wordData.top3 : [];
		}

		if (typeof liveConfidence === "number" && liveConfidence >= 0.55) {
			lastGoodPredictionRef.current = livePrediction || "---";
			setTranslatedText(livePrediction || "---");
			setConfidence(liveConfidence);
			setTop3(liveTop3);
		} else {
			setTranslatedText(lastGoodPredictionRef.current);
		}
	};

	const handleEnglish = async (processedPoints) => {
		const data = await fetchPrediction({
			language: "english",
			landmarks: processedPoints,
		});

		const nextPrediction = data?.prediction
			? String(data.prediction).toLowerCase()
			: "---";
		const nextConfidence =
			typeof data?.confidence === "number" ? data.confidence : null;

		if (nextConfidence !== null && nextConfidence >= LETTER_MIN_CONFIDENCE) {
			lastGoodPredictionRef.current = nextPrediction;
			setTranslatedText(nextPrediction);
			setConfidence(nextConfidence);
			setTop3(Array.isArray(data?.top3) ? data.top3 : []);
		} else {
			setTranslatedText(lastGoodPredictionRef.current);
		}

		tryAutoAppendEnglishLetter(nextPrediction, nextConfidence, data.top3);
	};

	// ── FIX 2: Arabic words direct mode — require only WORDS_SEQUENCE_LENGTH (3)
	//    frames and a lower confidence threshold (0.65) for faster commits.
	const handleArabicWordsDirect = async () => {
		if (sequenceFramesRef.current.length < 1) return;

		const data = await fetchPrediction({
			language: "arabic_words",
			sequence_landmarks: sequenceFramesRef.current,
		});

		// Wait until we have at least the minimum sequence length before committing
		if (sequenceFramesRef.current.length < WORDS_SEQUENCE_LENGTH && !data?.early_exit) return;

		const nextPrediction = data?.prediction ? String(data.prediction) : "---";
		const nextConfidence =
			typeof data?.confidence === "number" ? data.confidence : null;

		if (nextConfidence !== null && nextConfidence >= WORDS_MIN_CONFIDENCE) {
			lastGoodPredictionRef.current = nextPrediction;
			setTranslatedText(nextPrediction);
			setConfidence(nextConfidence);
			setTop3(Array.isArray(data?.top3) ? data.top3 : []);
		} else {
			setTranslatedText(lastGoodPredictionRef.current);
		}

		tryAutoAppendArabicWordDirect(nextPrediction, nextConfidence);
	};

	const sendLandmarksToBackend = async (points) => {
		const now = Date.now();
		if (now - lastSentTimeRef.current < REQUEST_INTERVAL_MS) return;
		lastSentTimeRef.current = now;

		try {
			const processedPoints = pushFrameToSequence(points);

			if (!processedPoints || processedPoints.length !== 63) {
				setError("Invalid hand landmarks format.");
				return;
			}

			setError("");

			if (lang === "arabic") {
				await handleArabicHybrid(processedPoints);
				return;
			}

			if (lang === "english") {
				await handleEnglish(processedPoints);
				return;
			}

			if (lang === "arabic_words") {
				await handleArabicWordsDirect();
			}
		} catch (err) {
			console.error("Fetch error:", err);
			setError(err.message || "Could not connect to backend.");
		}
	};

	const handleNoHandDetected = () => {
		const now = Date.now();

		if (!noHandSinceRef.current) {
			noHandSinceRef.current = now;
			spaceAddedRef.current = false;
			return;
		}

		const duration = now - noHandSinceRef.current;

		if (spaceAddedRef.current) return;

		if (lang === "arabic") {
			if (duration < ARABIC_NO_HAND_FINALIZE_MS) return;

			const hasLetters = arabicSegmentLettersRef.current.length > 0;
			const bestWord = getBestArabicWordStats();
			const hasWord =
				bestWord &&
				bestWord.count >= 3 &&
				bestWord.avgConf >= 0.82;

			if (hasLetters || hasWord) {
				finalizeArabicSegment();
			} else {
				resetArabicHybridTracking();
			}

			spaceAddedRef.current = true;
			return;
		}

		if (duration < NO_HAND_DELAY_MS) return;

		setBuiltTextSynced((prev) => {
			if (!prev || prev.endsWith(" ")) return prev;
			return `${prev} `;
		});

		spaceAddedRef.current = true;

		if (lang === "english") resetEnglishTracking();
		if (lang === "arabic_words") resetArabicWordsTracking();
	};

	const drawResults = (results) => {
		const canvas = canvasRef.current;
		const video = videoRef.current;
		if (!canvas || !video) return;

		const ctx = canvas.getContext("2d");
		canvas.width = video.videoWidth || 640;
		canvas.height = video.videoHeight || 480;
		ctx.clearRect(0, 0, canvas.width, canvas.height);

		if (results.multiHandLandmarks) {
			for (const landmarkList of results.multiHandLandmarks) {
				window.drawConnectors(ctx, landmarkList, window.HAND_CONNECTIONS, {
					color: "#38bdf8",
					lineWidth: 2,
				});
				window.drawLandmarks(ctx, landmarkList, {
					color: "#ffffff",
					lineWidth: 1,
					radius: 4,
				});
			}
		}
	};

	const startCamera = async () => {
		if (cameraOn) return;

		try {
			setError("");

			const stream = await navigator.mediaDevices.getUserMedia({
				video: true,
				audio: false,
			});

			streamRef.current = stream;

			if (videoRef.current) {
				videoRef.current.srcObject = stream;
			}

			await loadMediaPipe();

			const hands = new window.Hands({
				locateFile: (file) =>
					`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
			});

			hands.setOptions({
				maxNumHands: 1,
				modelComplexity: 1,
				minDetectionConfidence: 0.5,
				minTrackingConfidence: 0.4,
			});

			hands.onResults((results) => {
				drawResults(results);

				if (
					results.multiHandLandmarks &&
					results.multiHandLandmarks.length > 0
				) {
					setHandDetected(true);
					noHandSinceRef.current = null;
					spaceAddedRef.current = false;
					sendLandmarksToBackend(results.multiHandLandmarks[0]);
				} else {
					handleNoHandDetected();

					const now = Date.now();
					if (!noHandSinceRef.current) noHandSinceRef.current = now;

					if (now - noHandSinceRef.current >= NO_HAND_DELAY_MS) {
						setHandDetected(false);
						setTranslatedText("---");
						setConfidence(null);
						setTop3([]);
						setIsUnsure(false);
					}
				}
			});

			handsRef.current = hands;

			const camera = new window.Camera(videoRef.current, {
				onFrame: async () => {
					if (handsRef.current && videoRef.current) {
						await handsRef.current.send({ image: videoRef.current });
					}
				},
				width: 640,
				height: 480,
			});

			await camera.start();
			cameraUtilsRef.current = camera;
			setCameraOn(true);
		} catch (err) {
			if (err.name === "NotAllowedError") {
				setError("Camera permission was denied.");
			} else if (err.name === "NotFoundError") {
				setError("No camera was found on this device.");
			} else if (err.name === "NotReadableError") {
				setError("Camera is being used by another application.");
			} else {
				setError(`Could not access the camera: ${err.message}`);
			}

			resetPredictionState();
			setCameraOn(false);
		}
	};

	const stopCamera = () => {
		cameraUtilsRef.current?.stop?.();
		cameraUtilsRef.current = null;

		streamRef.current?.getTracks().forEach((track) => track.stop());
		streamRef.current = null;

		if (videoRef.current) videoRef.current.srcObject = null;

		const canvas = canvasRef.current;
		if (canvas) {
			const ctx = canvas.getContext("2d");
			ctx.clearRect(0, 0, canvas.width, canvas.height);
		}

		setError("");
		setCameraOn(false);
		resetPredictionState();
		resetEnglishTracking();
		resetArabicWordsTracking();
		resetArabicHybridTracking();
		noHandSinceRef.current = null;
		spaceAddedRef.current = false;
	};

	useEffect(() => {
		return () => {
			stopCamera();
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	if (!isValidLanguage) {
		return (
			<div className="page">
				<Navbar />
				<div className="translator-page">
					<div className="translator-container">
						<div className="translator-header">
							<h1>Language Not Supported</h1>
							<p>Please go back and choose a valid sign language.</p>
						</div>
						<Link to="/" className="back-link">
							← Back to Language Selection
						</Link>
					</div>
				</div>
				<Footer />
			</div>
		);
	}

	return (
		<div className="page">
			<Navbar />
			<div className="translator-page">
				<div className="translator-container">
					<div className="translator-header">
						<h1>Live Translator</h1>
						<p>Camera + MediaPipe + Flask + AI Model</p>
					</div>

					<div className="translator-layout">
						<div className="camera-box">
							<h2>Camera Feed</h2>

							<div className="camera-placeholder">
								<video
									ref={videoRef}
									autoPlay
									playsInline
									muted
									style={{
										position: "absolute",
										width: "100%",
										height: "100%",
										borderRadius: "18px",
										objectFit: "cover",
										display: cameraOn ? "block" : "none",
										transform: "scaleX(-1)",
									}}
								/>

								<canvas
									ref={canvasRef}
									style={{
										position: "absolute",
										width: "100%",
										height: "100%",
										borderRadius: "18px",
										objectFit: "cover",
										display: cameraOn ? "block" : "none",
										pointerEvents: "none",
										transform: "scaleX(-1)",
									}}
								/>

								{!cameraOn && <span>Camera will appear here</span>}
							</div>

							<div className="button-group">
								<button
									className="btn btn-primary"
									onClick={startCamera}
									disabled={cameraOn}
								>
									Start Camera
								</button>

								<button
									className="btn btn-secondary"
									onClick={stopCamera}
									disabled={!cameraOn}
								>
									Stop Camera
								</button>
							</div>

							{error && <p className="error-text">{error}</p>}
						</div>

						<div className="result-box">
							<h2>Translation Details</h2>

							<div className="result-item">
								<div className="result-label">Selected Language</div>
								<div className="result-value">{getLanguageName()}</div>
							</div>

							<div className="result-item">
								<div className="result-label">Dataset / Model</div>
								<div className="result-value">{getDatasetName()}</div>
							</div>

							<div className="result-item">
								<div className="result-label">Hand Detection</div>
								<div
									className="result-value"
									style={{ color: handDetected ? "#4ade80" : "#94a3b8" }}
								>
									{handDetected ? "✋ Hand Detected!" : "No hand detected"}
								</div>
							</div>

							<div className="result-item">
								<div className="result-label">Live Prediction</div>
								<div className="result-value">
									{translatedText}
									{isUnsure && handDetected && lang !== "arabic_words" && (
										<span
											style={{
												marginLeft: "8px",
												fontSize: "0.75rem",
												color: "#facc15",
											}}
										>
											⚠ unsure
										</span>
									)}
								</div>
							</div>

							<div className="result-item">
								<div className="result-label">Confidence</div>
								<div
									className="result-value"
									style={{
										color:
											confidence !== null && confidence >= 0.6
												? "#4ade80"
												: "#facc15",
									}}
								>
									{confidence !== null
										? `${(confidence * 100).toFixed(2)}%`
										: "---"}
								</div>
							</div>

							{lang === "arabic" && (
								<div className="result-item">
									<div className="result-label">Current Segment</div>
									<div className="result-value">
										{currentSegmentPreview || "---"}
									</div>
								</div>
							)}

							{top3.length > 0 && (
								<div className="result-item">
									<div className="result-label">Top 3 Predictions</div>
									<div className="result-value">
										{top3.map((item, idx) => {
											const shownValue =
												lang === "english"
													? String(
														item.prediction ||
														item.character ||
														item.label_name
													).toLowerCase()
													: String(
														item.prediction ||
														item.character ||
														item.label_name
													);

											return (
												<div
													key={idx}
													style={{ fontSize: "0.85rem", marginTop: "4px" }}
												>
													<span
														style={{
															fontWeight: idx === 0 ? "bold" : "normal",
														}}
													>
														{shownValue}
													</span>
													{" — "}
													<span
														style={{
															color:
																idx === 0 &&
																	top3[0].score - (top3[1]?.score || 0) <
																	LETTER_TOP3_MARGIN
																	? "#facc15"
																	: idx === 0
																		? "#4ade80"
																		: "#94a3b8",
														}}
													>
														{(item.score * 100).toFixed(1)}%
													</span>
												</div>
											);
										})}
									</div>
								</div>
							)}

							<div className="result-item">
								<div className="result-label">
									{lang === "arabic_words" ? "Built Sentence" : "Built Text"}
								</div>
								<div
									className="result-value"
									style={{
										transition: "background-color 0.3s ease",
										backgroundColor: flashCapture
											? "#4ade8033"
											: "transparent",
										borderRadius: "6px",
										padding: "2px 6px",
									}}
								>
									{getDisplayedBuiltText()}
								</div>
							</div>

							<div className="button-group" style={{ marginTop: "16px" }}>
								<button className="btn btn-secondary" onClick={addSpace}>
									Space
								</button>
								<button
									className="btn btn-secondary"
									onClick={deleteLastCharacter}
								>
									Delete
								</button>
								<button className="btn btn-secondary" onClick={clearBuiltText}>
									Clear
								</button>
								{lang === "arabic" && (
									<button
										className="btn btn-secondary"
										onClick={toggleArabicMode}
									>
										Mode:{" "}
										{manualArabicMode === "auto"
											? "Auto"
											: manualArabicMode === "letters"
												? "Letters"
												: "Words"}
									</button>
								)}
							</div>

							<Link to="/" className="back-link">
								← Back to Language Selection
							</Link>
						</div>
					</div>
				</div>
			</div>
			<Footer />
		</div>
	);
}

export default Translator;
