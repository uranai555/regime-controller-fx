#property strict

input string SourceId = "broker_a";
input string CanonicalSymbol = "XAUUSD";
input int FlushEveryN = 256;
input string OutputPrefix = "microstructure";

int g_handle = INVALID_HANDLE;
ulong g_sequence = 0;
int g_since_flush = 0;

#define SCHEMA_VERSION 1
#define RECORD_SIZE 84

bool WriteByte(int value) { return FileWriteInteger(g_handle, value, CHAR_VALUE) == 1; }
bool WriteU16(int value) { return FileWriteInteger(g_handle, value, SHORT_VALUE) == 2; }
bool WriteU32(uint value) { return FileWriteInteger(g_handle, (int)value, INT_VALUE) == 4; }
bool WriteI32(int value) { return FileWriteInteger(g_handle, value, INT_VALUE) == 4; }
bool WriteI64(long value) { return FileWriteLong(g_handle, value) == 8; }
bool WriteF64(double value) { return FileWriteDouble(g_handle, value) == 8; }

bool WriteHeader()
{
   return WriteByte('R') && WriteByte('F') && WriteByte('X') && WriteByte('4')
       && WriteU16(SCHEMA_VERSION) && WriteU16(RECORD_SIZE);
}

string SafeSourceId(string value)
{
   string out = "";
   for(int i=0; i<StringLen(value); i++)
   {
      ushort ch = StringGetCharacter(value, i);
      if((ch>='a' && ch<='z') || (ch>='A' && ch<='Z') || (ch>='0' && ch<='9') || ch=='_' || ch=='-')
         out += ShortToString(ch);
   }
   if(StringLen(out)==0) out = "source";
   return out;
}

int OnInit()
{
   if(FlushEveryN <= 0) return INIT_PARAMETERS_INCORRECT;
   string ts = TimeToString(TimeLocal(), TIME_DATE|TIME_MINUTES);
   StringReplace(ts, ".", "");
   StringReplace(ts, ":", "");
   StringReplace(ts, " ", "_");
   string file_name = OutputPrefix + "_" + SafeSourceId(SourceId) + "_" + CanonicalSymbol + "_" + ts + ".bin";
   ResetLastError();
   g_handle = FileOpen(file_name, FILE_WRITE|FILE_BIN|FILE_COMMON);
   if(g_handle == INVALID_HANDLE)
   {
      PrintFormat("QuoteCaptureEA: FileOpen failed err=%d", GetLastError());
      return INIT_FAILED;
   }
   if(!WriteHeader())
   {
      PrintFormat("QuoteCaptureEA: header write failed err=%d", GetLastError());
      FileClose(g_handle);
      g_handle = INVALID_HANDLE;
      return INIT_FAILED;
   }
   FileFlush(g_handle);
   PrintFormat("QuoteCaptureEA passive capture started source=%s symbol=%s file=%s", SourceId, CanonicalSymbol, file_name);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(g_handle != INVALID_HANDLE)
   {
      FileFlush(g_handle);
      FileClose(g_handle);
      g_handle = INVALID_HANDLE;
   }
}

void OnTick()
{
   if(g_handle == INVALID_HANDLE) return;

   MqlTick tick;
   if(!SymbolInfoTick(Symbol(), tick))
   {
      PrintFormat("QuoteCaptureEA: SymbolInfoTick failed err=%d", GetLastError());
      return;
   }

   uint host_ms = GetTickCount();
   ulong program_us = GetMicrosecondCount();
   double point = MarketInfo(Symbol(), MODE_POINT);
   int digits = (int)MarketInfo(Symbol(), MODE_DIGITS);

   bool ok = true;
   ok = ok && WriteI64((long)g_sequence);
   ok = ok && WriteU32(host_ms);
   ok = ok && WriteI64((long)program_us);
   ok = ok && WriteI64((long)tick.time);
   ok = ok && WriteI64((long)TimeLocal());
   ok = ok && WriteF64(tick.bid);
   ok = ok && WriteF64(tick.ask);
   ok = ok && WriteF64(tick.last);
   ok = ok && WriteI64((long)tick.volume);
   ok = ok && WriteF64(point);
   ok = ok && WriteI32(digits);
   ok = ok && WriteU32(0);

   if(!ok)
   {
      PrintFormat("QuoteCaptureEA: record write failed seq=%I64u err=%d", g_sequence, GetLastError());
      ExpertRemove();
      return;
   }

   g_sequence++;
   g_since_flush++;
   if(g_since_flush >= FlushEveryN)
   {
      FileFlush(g_handle);
      g_since_flush = 0;
   }
}
