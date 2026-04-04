# Frontend Documentation

The frontend is a React TypeScript Single Page Application (SPA) built with Vite for fast development and optimized production builds.

## Quick Start

```bash
cd frontend
npm install
npm run dev
```

App available at: http://localhost:3000  
Backend API: http://localhost:8000

## Architecture

### Core Technologies

- **React 18** - UI framework with hooks
- **TypeScript** - Static typing for JavaScript
- **Vite** - Lightning-fast build tool
- **Zustand** - Lightweight state management
- **Axios** - HTTP client
- **html2canvas** - Screenshot capture
- **React Google Maps API** - Maps integration (ready)

### Project Structure

```
src/
├── types/                 # TypeScript type definitions
├── services/              # API client utilities
├── hooks/                 # Custom React hooks
├── components/            # Reusable UI components
├── pages/                 # Feature-level components
├── App.tsx               # Main application
├── main.tsx              # React entry point
└── index.css             # Global styles
```

## State Management

Uses **Zustand** for lightweight, simple state management:

```typescript
import { useDetectionStore } from './hooks/useDetectionState'

// In a component:
const selectedModel = useDetectionStore((state) => state.selectedModel)
const setSelectedModel = useDetectionStore((state) => state.setSelectedModel)
const detecting = useDetectionStore((state) => state.detecting)
```

### Global State Structure

```typescript
{
  // Model Management
  availableModels: ModelsListResponse | null
  selectedModel: 'rubbish_area' | 'rubbish_classification' | 'traffic_sign'
  modelsLoading: boolean
  modelsError: string | null

  // Current Detection
  currentImage: string | null              // base64 encoded
  currentImageSize: { width, height } | null
  detectionResult: DetectionResult | null
  detecting: boolean                       // Loading state
  detectionError: string | null

  // UI State
  activeTab: 'video' | 'map'
  showHistory: boolean
}
```

## Custom Hooks

### useVideoCapture

Manages video element and frame capture:

```typescript
const {
  videoRef,           // Ref to <video> element
  canvasRef,          // Ref to hidden canvas
  isPlaying,          // Boolean state
  currentFrameDataUrl, // Base64 of last frame
  playVideo,          // () => void
  pauseVideo,         // () => void
  stopVideo,          // () => void
  captureFrame,       // () => VideoFrame | null
  loadVideo,          // (file: File) => void
} = useVideoCapture()
```

### useMapCapture

Manages map screenshot capture:

```typescript
const {
  mapContainerRef,     // Ref to container to capture
  isCapturing,         // Boolean state
  currentCaptureDataUrl, // Base64 of last capture
  captureMapView,      // () => Promise<MapFrame | null>
} = useMapCapture()
```

### useDetectionStore

Global state store (see State Management above).

## Components

### ModelSelector

Dropdown for selecting detection model.

**Props:**
```typescript
{
  models: ModelInfo[]    // Available models list
  disabled?: boolean     // Disable dropdown
}
```

**Usage:**
```tsx
<ModelSelector models={availableModels.models} disabled={detecting} />
```

### DetectionResults

Canvas overlay with bounding boxes and detection summary.

**Props:**
```typescript
{
  imageDataUrl: string          // Base64 image
  detections: DetectionBox[]    // Detection results
  imageWidth: number            // Image width
  imageHeight: number           // Image height
  inferenceTime: number         // Time in ms
}
```

**Features:**
- Draws bounding boxes on canvas
- Labels with confidence scores
- Color-coded by class
- Summary table of detections

### LoadingSpinner

Animated loading indicator.

**Props:**
```typescript
{
  message?: string              // Loading text
  size?: 'small' | 'medium' | 'large'  // Spinner size
}
```

### ErrorMessage

Dismissible error alert.

**Props:**
```typescript
{
  error: string                 // Error text
  onDismiss?: () => void       // Dismiss callback
}
```

## Pages

### VideoUploader

Handles video file selection with drag-and-drop.

**Features:**
- Drag-and-drop area
- File input click
- Format validation (.mp4, .avi, .mov, .webm)
- Size validation (max 500MB)
- User-friendly messages

**Usage:**
```tsx
<VideoUploader
  onVideoSelected={(file) => setVideoFile(file)}
  disabled={detecting}
/>
```

### VideoPlayer

Video playback with frame capture button.

**Features:**
- HTML5 video player with controls
- Play/Pause/Stop buttons
- Capture frame button
- Frame preview
- Automatic detection after capture

**Usage:**
```tsx
<VideoPlayer videoFile={videoFile} />
```

### MapViewer

Map display and view capture (Google Maps integration ready).

**Features:**
- Map container placeholder
- Capture view button
- Map preview
- Automatic detection after capture

**Usage:**
```tsx
<MapViewer />
```

## API Client Service

Located in `src/services/apiClient.ts`:

### DetectionService

```typescript
await DetectionService.detectFromBase64(imageBase64, modelType)
await DetectionService.detectFromFile(file, modelType)
```

### ModelService

```typescript
await ModelService.listModels()
```

### HistoryService

```typescript
await HistoryService.getHistory(modelType?, limit, offset)
await HistoryService.getResult(resultId)
```

### HealthService

```typescript
await HealthService.check()
```

## Styling

### Global Styles (`index.css`)
- Button styles
- Input styles
- Select/dropdown styles
- Focus states
- Disabled states

### Component Styles
Each component has its own .css file in the same directory:
- `ModelSelector.css`
- `DetectionResults.css`
- `LoadingSpinner.css`
- `ErrorMessage.css`
- `VideoUploader.css`
- `VideoPlayer.css`
- `MapViewer.css`

### App-level Styles (`App.css`)
- Layout and responsive design
- Header and footer
- Tab navigation
- Animations (fade-in)

## TypeScript Types

Located in `src/types/detection.ts`:

```typescript
interface DetectionBox {
  x_min: number
  y_min: number
  x_max: number
  y_max: number
  label: string
  confidence: number
}

interface DetectionResult {
  id: number
  model_type: string
  detections: DetectionBox[]
  image_height: number
  image_width: number
  inference_time_ms: number
  created_at: string
}

type ModelType = 'rubbish_area' | 'rubbish_classification' | 'traffic_sign'
```

## Configuration

### Build Configuration (`vite.config.ts`)

```typescript
{
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {            // Proxy API requests to backend
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '/api'),
      },
    },
  },
}
```

### TypeScript Configuration (`tsconfig.json`)

- Target: ES2020
- Module: ESNext
- Strict mode enabled
- JSX: react-jsx

## Development

### Development Commands

```bash
# Start dev server with hot reload
npm run dev

# Type check without building
npm run type-check

# Build for production
npm run build

# Preview production build locally
npm run preview

# Lint code (ESLint)
npm run lint
```

### Debugging

**Browser DevTools:**
- Open F12 in browser
- Network tab: Monitor API calls to backend
- Console: Check for JavaScript errors
- React DevTools: Inspect component hierarchy

**Zustand DevTools:**
Can integrate Redux DevTools for state inspection:
```javascript
// Add to useDetectionStore for debugging
middleware: (f) => devtools(f)
```

## Performance Optimization

### Code Splitting
- Routes can be lazy loaded with `React.lazy()`
- Components only loaded when needed

### Memoization
- Use `React.memo()` for expensive components
- Use `useCallback()` for event handlers

### Image Optimization
- JPEG compression (95% quality) before sending to backend
- Canvas rendering for efficient drawing

### Build Optimization
- Vite automatically chunks dependencies
- Tree-shaking removes unused code
- CSS modules can be used for scoped styling

## Error Handling

**API Errors:**
- Try-catch blocks in async operations
- Errors stored in Zustand state
- ErrorMessage component displays to user

**Input Validation:**
- Video format/size validation
- Image format/size validation
- Model selection validation

## Environment Variables

Create `.env.local` for local development:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT=30000
```

Access in code:
```typescript
const apiUrl = import.meta.env.VITE_API_BASE_URL
```

## Testing

Framework placeholder for future tests. Can add:
- Jest - Unit testing
- React Testing Library - Component testing
- Cypress - E2E testing

```bash
npm install --save-dev jest @testing-library/react @testing-library/jest-dom
```

## Production Build

```bash
npm run build
```

Creates optimized files in `dist/` directory:
- Minified JavaScript
- Bundled CSS
- Asset optimization
- Source maps for debugging

Serve with:
```bash
npm run preview  # Local preview
# Or deploy dist/ folder to any static host
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Common Issues

**API Connection Error**
- Check backend running on http://localhost:8000
- Check browser console for CORS errors
- Verify `VITE_API_BASE_URL` in .env.local

**Video Won't Play**
- Check format support (MP4, AVI, MOV, WebM)
- Try different video codec
- Check browser compatibility

**Models Not Loading**
- Check `/api/models` endpoint returns data
- Check browser Network tab for API calls
- Verify backend is running

## Resources

- [React Documentation](https://react.dev)
- [TypeScript Handbook](https://www.typescriptlang.org/docs)
- [Vite Documentation](https://vitejs.dev)
- [Zustand GitHub](https://github.com/pmndrs/zustand)
- [Axios Documentation](https://axios-http.com)

---

**Frontend Version**: 1.0.0  
**React Version**: 18.2.0  
**TypeScript Version**: 5.2.2  
**Vite Version**: 5.0.8  
**Last Updated**: April 2026
