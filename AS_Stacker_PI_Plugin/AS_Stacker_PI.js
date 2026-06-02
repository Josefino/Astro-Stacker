// Astro Stacker PixInsight wrapper
// Runs astro_stacker_cli.py and writes FITS outputs with AS_ prefix.

#include <pjsr/StdButton.jsh>
#include <pjsr/StdDialogCode.jsh>
#include <pjsr/FrameStyle.jsh>
#include <pjsr/Sizer.jsh>
#include <pjsr/TextAlign.jsh>
#include <pjsr/DataType.jsh>
#include <pjsr/NumericControl.jsh>

#feature-id    Utilities > AS_Stacker
#feature-info  Astro Stacker wrapper for PixInsight. Calls the external Python CLI engine.

var AS_STACKER_SETTINGS_ID = "AS_Stacker/settings";
var AS_STACKER_SCRIPT_DIR = File.extractDrive( #__FILE__ ) + File.extractDirectory( #__FILE__ );
if ( AS_STACKER_SCRIPT_DIR.length > 0 && AS_STACKER_SCRIPT_DIR[AS_STACKER_SCRIPT_DIR.length - 1] != '/' )
   AS_STACKER_SCRIPT_DIR += "/";

function defaultPython()
{
   if ( CoreApplication.platform == "Windows" )
      return "python";
   return "python3";
}

function bundledCliPath()
{
   return AS_STACKER_SCRIPT_DIR + "astro_stacker_cli.py";
}

function defaultSettings()
{
   return {
      python: defaultPython(),
      cli: bundledCliPath(),
      input: "",
      outputDir: "",
      outputName: "AS_stack.fit",
      align: 0,
      stack: 0,
      bayer: 0,
      sigma: 2.5,
      maxImages: 0,
      keepPercent: 80,
      maxShift: 180,
      border: 120,
      processes: 1,
      rawOnly: false,
      autoRef: true,
      quality: false,
      strictStars: true,
      normalize: true,
      mosaic: false,
      gpu: false
   };
}

function loadSettings()
{
   var settings = defaultSettings();
   try
   {
      var text = Settings.read( AS_STACKER_SETTINGS_ID, DataType_String );
      if ( text != null && text.length > 0 )
      {
         var saved = JSON.parse( text );
         for ( var key in saved )
            settings[key] = saved[key];
      }
   }
   catch ( e )
   {
   }
   if ( !fileExists( settings.cli ) && fileExists( bundledCliPath() ) )
      settings.cli = bundledCliPath();
   return settings;
}

function saveSettings( settings )
{
   try
   {
      Settings.write( AS_STACKER_SETTINGS_ID, DataType_String, JSON.stringify( settings ) );
   }
   catch ( e )
   {
   }
}

function asPrefixName( name )
{
   if ( name.length == 0 )
      name = "AS_stack.fit";
   if ( name.indexOf( ".fit" ) < 0 && name.indexOf( ".fits" ) < 0 )
      name += ".fit";
   if ( name.substring( 0, 3 ) != "AS_" )
      name = "AS_" + name;
   return name;
}

function fileExists( path )
{
   try
   {
      return File.exists( path );
   }
   catch ( e )
   {
      return false;
   }
}

function dirExists( path )
{
   try
   {
      return File.directoryExists( path );
   }
   catch ( e )
   {
      return false;
   }
}

function removeFileIfExists( path )
{
   if ( !fileExists( path ) )
      return true;
   try
   {
      File.remove( path );
      return true;
   }
   catch ( e )
   {
      Console.warningln( "Could not remove previous output: " + path );
      Console.warningln( e.toString() );
      return false;
   }
}

function msleep( milliseconds )
{
   var start = new Date().getTime();
   while ( new Date().getTime() - start < milliseconds )
      processEvents();
}

function outputPathCandidates( outDir, outName )
{
   var paths = [ outDir + "/" + outName ];
   var lower = outName.toLowerCase();
   if ( lower.indexOf( ".fits" ) == lower.length - 5 )
      paths.push( outDir + "/" + outName.substring( 0, outName.length - 5 ) + ".fit" );
   else if ( lower.indexOf( ".fit" ) == lower.length - 4 )
      paths.push( outDir + "/" + outName.substring( 0, outName.length - 4 ) + ".fits" );
   return paths;
}

function newestRecentStackFile( outDir, notBeforeMs )
{
   var patterns = [ "AS_*.fit", "AS_*.fits" ];
   var bestPath = "";
   var bestTime = 0;
   var f = new FileFind;
   for ( var p = 0; p < patterns.length; ++p )
   {
      if ( f.begin( outDir + "/" + patterns[p] ) )
      {
         do
         {
            if ( f.isFile )
            {
               var path = outDir + "/" + f.name;
               try
               {
                  var modified = new FileInfo( path ).lastModified.getTime();
                  if ( modified >= notBeforeMs - 5000 && modified > bestTime )
                  {
                     bestPath = path;
                     bestTime = modified;
                  }
               }
               catch ( e )
               {
               }
            }
         }
         while ( f.next() );
         f.end();
      }
   }
   return bestPath;
}

function waitForStackOutput( outDir, outName, notBeforeMs, timeoutMs )
{
   var paths = outputPathCandidates( outDir, outName );
   var started = new Date().getTime();
   while ( new Date().getTime() - started <= timeoutMs )
   {
      for ( var i = 0; i < paths.length; ++i )
         if ( fileExists( paths[i] ) )
            return paths[i];

      var recent = newestRecentStackFile( outDir, notBeforeMs );
      if ( recent.length > 0 )
         return recent;

      msleep( 250 );
   }
   return "";
}

function chooseFile( caption, filters )
{
   var d = new OpenFileDialog;
   d.caption = caption;
   d.filters = filters;
   if ( d.execute() )
      return d.fileName;
   return "";
}

function chooseDirectory( caption )
{
   var d = new GetDirectoryDialog;
   d.caption = caption;
   if ( d.execute() )
      return d.directory;
   return "";
}

function openStackOutputWindow( path )
{
   try
   {
      var windows = ImageWindow.open( path );
      for ( var i = 0; i < windows.length; ++i )
      {
         windows[i].show();
         try
         {
            windows[i].bringToFront();
         }
         catch ( e2 )
         {
         }
      }
      Console.writeln( "Opened in PixInsight: " + path );
   }
   catch ( e )
   {
      Console.warningln( "Could not open output automatically: " + e.toString() );
   }
}

function savedPathFromLine( line )
{
   var match = line.match( /Saved:\s*(.+)$/ );
   if ( match == null )
      return "";
   return match[1].trim();
}

function createConsoleStreamHandler( onStdoutLine )
{
   var stdoutBuffer = "";
   var stderrBuffer = "";

   function flushLines( text, isError, force )
   {
      var buffer = isError ? stderrBuffer : stdoutBuffer;
      buffer += String( text );
      var lines = buffer.split( /\r?\n/ );

      if ( force )
         buffer = "";
      else
         buffer = lines.pop();

      var count = force ? lines.length : lines.length;
      for ( var i = 0; i < count; ++i )
      {
         var line = lines[i];
         if ( line.length == 0 )
            continue;
         if ( isError )
            Console.warningln( line );
         else
         {
            Console.writeln( line );
            if ( onStdoutLine )
               onStdoutLine( line );
         }
      }

      if ( isError )
         stderrBuffer = buffer;
      else
         stdoutBuffer = buffer;
   }

   return {
      stdout: function( text ) { flushLines( text, false, false ); },
      stderr: function( text ) { flushLines( text, true, false ); },
      finish: function()
      {
         if ( stdoutBuffer.length > 0 )
            Console.writeln( stdoutBuffer );
         if ( stderrBuffer.length > 0 )
            Console.warningln( stderrBuffer );
         stdoutBuffer = "";
         stderrBuffer = "";
      }
   };
}

function runExternalProcessLive( program, args, onStdoutLine )
{
   var process = new ExternalProcess;
   var handler = createConsoleStreamHandler( onStdoutLine );
   var started = false;
   var finished = false;
   var sawOutput = false;
   var sawProcessError = false;

   process.onStarted = function()
   {
      started = true;
      Console.writeln( "Started." );
   };

   process.onStandardOutputDataAvailable = function()
   {
      sawOutput = true;
      started = true;
      handler.stdout( String( this.stdout ) );
   };

   process.onStandardErrorDataAvailable = function()
   {
      sawOutput = true;
      started = true;
      handler.stderr( String( this.stderr ) );
   };

   process.onError = function( code )
   {
      sawProcessError = true;
   };

   process.onFinished = function()
   {
      finished = true;
      handler.finish();
      Console.writeln( "Process finished." );
   };

   try
   {
      if ( !process.start( program, args ) )
         return false;

      var waitStartedMs = new Date().getTime();
      while ( process.isStarting && !started )
         processEvents();

      while ( !started )
      {
         processEvents();
         if ( new Date().getTime() - waitStartedMs > 10000 )
         {
            handler.finish();
            if ( sawProcessError )
               Console.warningln( "External process reported an error before startup." );
            return false;
         }
      }

      while ( !finished )
      {
         processEvents();
         msleep( 50 );
      }

      handler.finish();
      return started && finished;
   }
   catch ( e )
   {
      handler.finish();
      Console.criticalln( e.toString() );
      return false;
   }
}

function labeledEdit( parent, labelText, text )
{
   var label = new Label( parent );
   label.text = labelText;
   label.textAlignment = TextAlign_Right | TextAlign_VertCenter;
   label.minWidth = parent.labelWidth;

   var edit = new Edit( parent );
   edit.text = text || "";
   edit.minWidth = 420;

   var sizer = new HorizontalSizer;
   sizer.spacing = 6;
   sizer.add( label );
   sizer.add( edit, 100 );

   return { label: label, edit: edit, sizer: sizer };
}

function browseButton( parent, onClick )
{
   var b = new PushButton( parent );
   b.text = "...";
   b.toolTip = "Browse";
   b.onClick = onClick;
   return b;
}

function ASStackerDialog()
{
   this.__base__ = Dialog;
   this.__base__();

   this.windowTitle = "AS_Stacker - PixInsight Wrapper";
   this.labelWidth = this.font.width( "Star border margin:" ) + 12;
   var saved = loadSettings();

   var py = labeledEdit( this, "Python:", saved.python );
   var cli = labeledEdit( this, "CLI script:", saved.cli );
   var input = labeledEdit( this, "Light folder:", saved.input );
   var outputDir = labeledEdit( this, "Output folder:", saved.outputDir );
   var outputName = labeledEdit( this, "Output name:", saved.outputName );

   var pyBrowse = browseButton( this, function()
   {
      var f = chooseFile( "Select Python executable", [["Executables", "*"]] );
      if ( f.length > 0 )
         py.edit.text = f;
   } );
   py.sizer.add( pyBrowse );

   var cliBrowse = browseButton( this, function()
   {
      var f = chooseFile( "Select astro_stacker_cli.py", [["Python scripts", "*.py"], ["All files", "*"]] );
      if ( f.length > 0 )
         cli.edit.text = f;
   } );
   cli.sizer.add( cliBrowse );

   var inputBrowse = browseButton( this, function()
   {
      var d = chooseDirectory( "Select light frames folder" );
      if ( d.length > 0 )
      {
         input.edit.text = d;
         if ( outputDir.edit.text.length == 0 )
            outputDir.edit.text = d + "/astro_stacker_output";
      }
   } );
   input.sizer.add( inputBrowse );

   var outputBrowse = browseButton( this, function()
   {
      var d = chooseDirectory( "Select output folder" );
      if ( d.length > 0 )
         outputDir.edit.text = d;
   } );
   outputDir.sizer.add( outputBrowse );

   this.alignCombo = new ComboBox( this );
   this.alignCombo.addItem( "Star alignment + RANSAC" );
   this.alignCombo.addItem( "Translation" );
   this.alignCombo.addItem( "ECC affine" );
   this.alignCombo.addItem( "Calibration/no alignment" );
   this.alignCombo.currentItem = Math.max( 0, Math.min( 3, saved.align ) );

   this.stackCombo = new ComboBox( this );
   this.stackCombo.addItem( "Sigma clip" );
   this.stackCombo.addItem( "Median" );
   this.stackCombo.addItem( "Mean" );
   this.stackCombo.addItem( "High rejection mean" );
   this.stackCombo.currentItem = Math.max( 0, Math.min( 3, saved.stack ) );

   this.bayerCombo = new ComboBox( this );
   this.bayerCombo.addItem( "Auto" );
   this.bayerCombo.addItem( "Mono" );
   this.bayerCombo.addItem( "RGGB" );
   this.bayerCombo.addItem( "BGGR" );
   this.bayerCombo.addItem( "GRBG" );
   this.bayerCombo.addItem( "GBRG" );
   this.bayerCombo.currentItem = Math.max( 0, Math.min( 5, saved.bayer ) );

   this.sigmaSpin = new NumericControl( this );
   this.sigmaSpin.label.text = "Sigma:";
   this.sigmaSpin.label.minWidth = this.labelWidth;
   this.sigmaSpin.setRange( 0.5, 8.0 );
   this.sigmaSpin.slider.setRange( 5, 80 );
   this.sigmaSpin.setPrecision( 2 );
   this.sigmaSpin.setValue( saved.sigma );

   this.maxImagesSpin = new SpinBox( this );
   this.maxImagesSpin.minValue = 0;
   this.maxImagesSpin.maxValue = 100000;
   this.maxImagesSpin.value = saved.maxImages;

   this.keepSpin = new SpinBox( this );
   this.keepSpin.minValue = 10;
   this.keepSpin.maxValue = 100;
   this.keepSpin.value = saved.keepPercent;

   this.maxShiftSpin = new SpinBox( this );
   this.maxShiftSpin.minValue = 20;
   this.maxShiftSpin.maxValue = 1000;
   this.maxShiftSpin.value = saved.maxShift;

   this.borderSpin = new SpinBox( this );
   this.borderSpin.minValue = 0;
   this.borderSpin.maxValue = 5000;
   this.borderSpin.value = saved.border;

   this.processesSpin = new SpinBox( this );
   this.processesSpin.minValue = 1;
   this.processesSpin.maxValue = 64;
   this.processesSpin.value = saved.processes;

   this.rawOnlyCheck = new CheckBox( this );
   this.rawOnlyCheck.text = "RAW only";
   this.rawOnlyCheck.toolTip = "Use only FIT/FITS and camera RAW files. JPG/PNG/BMP/TIFF previews are ignored, including in automatic Flat/Bias/Dark folders.";
   this.rawOnlyCheck.checked = saved.rawOnly;

   this.autoRefCheck = new CheckBox( this );
   this.autoRefCheck.text = "Auto reference";
   this.autoRefCheck.checked = saved.autoRef;

   this.qualityCheck = new CheckBox( this );
   this.qualityCheck.text = "Quality filter";
   this.qualityCheck.checked = saved.quality;

   this.strictStarsCheck = new CheckBox( this );
   this.strictStarsCheck.text = "Strict star filter";
   this.strictStarsCheck.checked = saved.strictStars;

   this.normalizeCheck = new CheckBox( this );
   this.normalizeCheck.text = "Normalize background";
   this.normalizeCheck.checked = saved.normalize;

   this.mosaicCheck = new CheckBox( this );
   this.mosaicCheck.text = "Mosaic canvas";
   this.mosaicCheck.toolTip = "Expand the output canvas to include all aligned frames. With GPU enabled, mosaic integration uses VRAM tiles; otherwise it uses the parallel CPU path.";
   this.mosaicCheck.checked = saved.mosaic;

   this.gpuCheck = new CheckBox( this );
   this.gpuCheck.text = "Use GPU";
   this.gpuCheck.checked = saved.gpu;

   function comboRow( dialog, text, control )
   {
      var label = new Label( dialog );
      label.text = text;
      label.textAlignment = TextAlign_Right | TextAlign_VertCenter;
      label.minWidth = dialog.labelWidth;
      var s = new HorizontalSizer;
      s.spacing = 6;
      s.add( label );
      s.add( control, 100 );
      return s;
   }

   function spinRow( dialog, text, control )
   {
      var label = new Label( dialog );
      label.text = text;
      label.textAlignment = TextAlign_Right | TextAlign_VertCenter;
      label.minWidth = dialog.labelWidth;
      var s = new HorizontalSizer;
      s.spacing = 6;
      s.add( label );
      s.add( control );
      s.addStretch();
      return s;
   }

   this.runButton = new PushButton( this );
   this.runButton.text = "Run";
   this.runButton.icon = this.scaledResource( ":/icons/power.png" );
   this.runButton.onClick = function()
   {
      var python = py.edit.text.trim();
      var cliPath = cli.edit.text.trim();
      var inputPath = input.edit.text.trim();
      var outDir = outputDir.edit.text.trim();
      var outName = asPrefixName( outputName.edit.text.trim() );

      if ( python.length == 0 )
      {
         new MessageBox( "Python executable is not set.", "AS_Stacker", StdIcon_Error, StdButton_Ok ).execute();
         return;
      }
      if ( !fileExists( cliPath ) )
      {
         new MessageBox( "CLI script does not exist:\n" + cliPath, "AS_Stacker", StdIcon_Error, StdButton_Ok ).execute();
         return;
      }
      if ( !dirExists( inputPath ) )
      {
         new MessageBox( "Light folder does not exist:\n" + inputPath, "AS_Stacker", StdIcon_Error, StdButton_Ok ).execute();
         return;
      }
      if ( outDir.length == 0 )
         outDir = inputPath + "/astro_stacker_output";
      var logPath = outDir + "/AS_stacker_cli_error.log";

      var alignValues = [ "star_affine", "translation", "ecc_affine", "calibration" ];
      var stackValues = [ "sigma", "median", "mean", "high_rejection" ];
      var bayerValues = [ "auto", "mono", "RGGB", "BGGR", "GRBG", "GBRG" ];

      var args = [
         cliPath,
         inputPath,
         "--output-dir", outDir,
         "--output-name", outName,
         "--align", alignValues[this.dialog.alignCombo.currentItem],
         "--stack", stackValues[this.dialog.stackCombo.currentItem],
         "--sigma", this.dialog.sigmaSpin.value.toString(),
         "--max-images", this.dialog.maxImagesSpin.value.toString(),
         "--keep-percent", this.dialog.keepSpin.value.toString(),
         "--max-star-shift", this.dialog.maxShiftSpin.value.toString(),
         "--star-border-margin", this.dialog.borderSpin.value.toString(),
         "--bayer", bayerValues[this.dialog.bayerCombo.currentItem],
         "--processes", this.dialog.processesSpin.value.toString()
      ];

      if ( this.dialog.gpuCheck.checked )
         args.push( "--gpu" );
      if ( this.dialog.rawOnlyCheck.checked )
         args.push( "--raw-only" );
      if ( !this.dialog.normalizeCheck.checked )
         args.push( "--no-normalize-background" );
      if ( !this.dialog.autoRefCheck.checked )
         args.push( "--no-auto-reference" );
      if ( this.dialog.qualityCheck.checked )
         args.push( "--quality-filter" );
      if ( !this.dialog.strictStarsCheck.checked )
         args.push( "--no-strict-star-filter" );
      if ( this.dialog.mosaicCheck.checked )
         args.push( "--mosaic" );

      saveSettings( {
         python: python,
         cli: cliPath,
         input: inputPath,
         outputDir: outDir,
         outputName: outName,
         align: this.dialog.alignCombo.currentItem,
         stack: this.dialog.stackCombo.currentItem,
         bayer: this.dialog.bayerCombo.currentItem,
         sigma: this.dialog.sigmaSpin.value,
         maxImages: this.dialog.maxImagesSpin.value,
         keepPercent: this.dialog.keepSpin.value,
         maxShift: this.dialog.maxShiftSpin.value,
         border: this.dialog.borderSpin.value,
         processes: this.dialog.processesSpin.value,
         rawOnly: this.dialog.rawOnlyCheck.checked,
         autoRef: this.dialog.autoRefCheck.checked,
         quality: this.dialog.qualityCheck.checked,
         strictStars: this.dialog.strictStarsCheck.checked,
         normalize: this.dialog.normalizeCheck.checked,
         mosaic: this.dialog.mosaicCheck.checked,
         gpu: this.dialog.gpuCheck.checked
      } );

      Console.show();
      Console.writeln( "<end><cbr><br><b>AS_Stacker</b>" );
      Console.writeln( "Python: " + python );
      Console.writeln( "CLI: " + cliPath );
      Console.writeln( "Input: " + inputPath );
      Console.writeln( "Output: " + outDir + "/" + outName );
      Console.writeln( "Running external process..." );

      var outputPath = outDir + "/" + outName;
      if ( !removeFileIfExists( outputPath ) )
      {
         new MessageBox( "The previous output file could not be removed:\n" + outputPath +
                         "\n\nClose it in PixInsight, choose another output name, or select another output folder.",
                         "AS_Stacker", StdIcon_Error, StdButton_Ok ).execute();
         return;
      }

      var openedOutput = false;
      runExternalProcessLive( python, args, function( line )
      {
         if ( openedOutput )
            return;
         var savedPath = savedPathFromLine( line );
         if ( savedPath.length == 0 )
            return;
         openedOutput = true;
         openStackOutputWindow( savedPath );
      } );
   };
   this.runButton.dialog = this;

   this.cancelButton = new PushButton( this );
   this.cancelButton.text = "Close";
   this.cancelButton.onClick = function() { this.dialog.cancel(); };
   this.cancelButton.dialog = this;

   this.info = new Label( this );
   this.info.text = "This is a PixInsight wrapper around astro_stacker_cli.py. Output files always start with AS_.";
   this.info.wordWrapping = true;
   this.info.frameStyle = FrameStyle_Box;
   this.info.margin = 6;

   this.sizer = new VerticalSizer;
   this.sizer.margin = 8;
   this.sizer.spacing = 6;
   this.sizer.add( this.info );
   this.sizer.add( py.sizer );
   this.sizer.add( cli.sizer );
   this.sizer.add( input.sizer );
   this.sizer.add( outputDir.sizer );
   this.sizer.add( outputName.sizer );
   this.sizer.add( comboRow( this, "Alignment:", this.alignCombo ) );
   this.sizer.add( comboRow( this, "Stacking:", this.stackCombo ) );
   this.sizer.add( comboRow( this, "Bayer FIT:", this.bayerCombo ) );
   this.sizer.add( this.sigmaSpin );
   this.sizer.add( spinRow( this, "Max frames:", this.maxImagesSpin ) );
   this.sizer.add( spinRow( this, "Keep %:", this.keepSpin ) );
   this.sizer.add( spinRow( this, "Max star drift:", this.maxShiftSpin ) );
   this.sizer.add( spinRow( this, "Ignore border:", this.borderSpin ) );
   this.sizer.add( spinRow( this, "CPU processes:", this.processesSpin ) );

   var checks = new HorizontalSizer;
   checks.spacing = 12;
   checks.add( this.rawOnlyCheck );
   checks.add( this.autoRefCheck );
   checks.add( this.qualityCheck );
   checks.add( this.strictStarsCheck );
   checks.add( this.normalizeCheck );
   checks.add( this.mosaicCheck );
   checks.add( this.gpuCheck );
   checks.addStretch();
   this.sizer.add( checks );

   var buttons = new HorizontalSizer;
   buttons.spacing = 6;
   buttons.addStretch();
   buttons.add( this.runButton );
   buttons.add( this.cancelButton );
   this.sizer.add( buttons );

   this.adjustToContents();
}
ASStackerDialog.prototype = new Dialog;

function main()
{
   var dialog = new ASStackerDialog;
   dialog.execute();
}

main();
