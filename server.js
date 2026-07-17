import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { createClient } from '@supabase/supabase-js';
import Stripe from 'stripe';
import Anthropic from '@anthropic-ai/sdk';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { z } from 'zod';

const app = express();
const PORT = process.env.PORT || 3001;

const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_SERVICE_KEY);
const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });

// ── CORS — allow all origins (fix for Vercel → Render) ──
app.use(cors({
  origin: '*',
  methods: ['GET','POST','PUT','DELETE','OPTIONS'],
  allowedHeaders: ['Content-Type','Authorization'],
}));
app.options('*', cors());

app.use(helmet({ crossOriginResourcePolicy: false }));
app.use(express.json({ limit: '1mb' }));

const authLimiter  = rateLimit({ windowMs: 15*60*1000, max: 50 });
const apiLimiter   = rateLimit({ windowMs: 60*1000, max: 100 });

function requireAuth(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  try { req.user = jwt.verify(token, process.env.JWT_SECRET); next(); }
  catch { res.status(401).json({ error: 'Invalid token' }); }
}

async function requirePro(req, res, next) {
  const { data: user } = await supabase.from('users').select('plan').eq('id', req.user.id).single();
  if (!user || user.plan !== 'pro') return res.status(403).json({ error: 'Pro plan required', upgrade: true });
  next();
}

const BusinessSchema = z.object({
  industry: z.string(), stage: z.string(),
  revenue: z.number().min(0), customers: z.number().min(1),
  avgPrice: z.number().min(0), fixedCosts: z.number().min(0),
  varCost: z.number().min(0), marketing: z.number().min(0),
  monthlyBurn: z.number().min(0), capital: z.number().min(0),
  growth: z.number().min(0).max(300),
  s1: z.number().min(0).max(100).optional(),
  s2: z.number().min(0).max(100).optional(),
  s3: z.number().min(0).max(100).optional(),
});

// ════════════════════════════════════
// AI PROXY — main endpoint for all AI
// ════════════════════════════════════
app.post('/api/ai/complete', apiLimiter, async (req, res) => {
  const { prompt, max_tokens = 1000 } = req.body;
  if (!prompt) return res.status(400).json({ error: 'prompt required' });
  try {
    const msg = await anthropic.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens,
      messages: [{ role: 'user', content: prompt }],
    });
    res.json({ text: msg.content[0].text });
  } catch (e) {
    console.error('AI error:', e.message);
    res.status(500).json({ error: 'AI unavailable', detail: e.message });
  }
});

// ════════════════════════════════════
// AUTH
// ════════════════════════════════════
app.post('/api/auth/register', authLimiter, async (req, res) => {
  const { email, password, name } = req.body;
  if (!email || !password || !name) return res.status(400).json({ error: 'Missing fields' });
  if (password.length < 8) return res.status(400).json({ error: 'Password too short' });
  const hash = await bcrypt.hash(password, 12);
  const { data, error } = await supabase.from('users').insert({
    email: email.toLowerCase(), password_hash: hash, name, plan: 'free', analyses_count: 0,
  }).select('id, email, name, plan').single();
  if (error) {
    if (error.code === '23505') return res.status(409).json({ error: 'Email already exists' });
    return res.status(500).json({ error: 'Registration failed' });
  }
  const token = jwt.sign({ id: data.id, email: data.email }, process.env.JWT_SECRET, { expiresIn: '7d' });
  res.json({ token, user: { id: data.id, email: data.email, name: data.name, plan: data.plan } });
});

app.post('/api/auth/login', authLimiter, async (req, res) => {
  const { email, password } = req.body;
  const { data: user } = await supabase.from('users').select('*').eq('email', email.toLowerCase()).single();
  if (!user || !(await bcrypt.compare(password, user.password_hash)))
    return res.status(401).json({ error: 'Invalid credentials' });
  const token = jwt.sign({ id: user.id, email: user.email }, process.env.JWT_SECRET, { expiresIn: '7d' });
  res.json({ token, user: { id: user.id, email: user.email, name: user.name, plan: user.plan } });
});

app.get('/api/auth/me', requireAuth, async (req, res) => {
  const { data } = await supabase.from('users').select('id,email,name,plan,analyses_count,created_at').eq('id', req.user.id).single();
  res.json(data);
});

// ════════════════════════════════════
// ANALYSIS
// ════════════════════════════════════
function computeMetrics(d) {
  const cm=d.avgPrice-d.varCost, cmPct=d.avgPrice>0?(cm/d.avgPrice)*100:0;
  const tvc=d.varCost*d.customers, tc=d.fixedCosts+tvc+d.marketing;
  const np=d.revenue-tc, gm=d.revenue>0?((d.revenue-tvc)/d.revenue)*100:0;
  const om=d.revenue>0?(np/d.revenue)*100:0, fcb=tc>0?(d.fixedCosts/tc)*100:0;
  const beu=cm>0?Math.ceil(d.fixedCosts/cm):0, ber=beu*d.avgPrice;
  const mbe=(d.growth>0&&beu>d.customers)?Math.ceil((beu-d.customers)/(d.customers*d.growth/100)):0;
  const rpc=d.customers>0?d.revenue/d.customers:0;
  const ltv=(d.avgPrice*12)*(cmPct/100), cac=d.avgPrice*0.4;
  const ltvCac=cac>0?ltv/cac:0;
  const burn=np<0?Math.abs(np):d.monthlyBurn, runway=burn>0?d.capital/burn:999;
  const oli=(d.revenue-d.fixedCosts)!==0?d.revenue/(d.revenue-d.fixedCosts):0;
  let cr='N/A',cl='gray';
  if(d.s1!==undefined){const mx=Math.max(d.s1||0,d.s2||0,d.s3||0);cr=mx>=70?'HIGH':mx>=50?'MEDIUM':'LOW';cl=mx>=70?'red':mx>=50?'orange':'green';}
  const grs=Math.min(100,Math.max(0,Math.round((gm>60?20:gm>30?12:6)+(ltvCac>3?20:ltvCac>1?12:5)+(runway>12?20:runway>6?12:4)+(om>15?20:om>0?12:0)+(cr==='LOW'?20:cr==='MEDIUM'?10:0))));
  const ss=Math.min(100,Math.max(0,Math.round((gm*0.25)+(ltvCac*5)+(om*0.5)+(runway>6?20:runway*3)+(cr==='LOW'?5:cr==='MEDIUM'?2:0))));
  return {np,gm,om,fcb,cm,cmPct,tvc,tc,beu,ber,mbe,rpc,ltv,cac,ltvCac,burn,runway,oli,cr,cl,grs,ss};
}

app.post('/api/analyse/basic', requireAuth, apiLimiter, async (req, res) => {
  const parsed = BusinessSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: 'Invalid data' });
  const calc = computeMetrics(parsed.data);
  const { data: analysis } = await supabase.from('analyses').insert({
    user_id: req.user.id, business_data: parsed.data, metrics: calc, tier: 'basic',
  }).select('id').single();
  res.json({ analysisId: analysis?.id, metrics: calc });
});

app.get('/api/analyses', requireAuth, async (req, res) => {
  const { data } = await supabase.from('analyses').select('id,created_at,tier,metrics,business_data').eq('user_id', req.user.id).order('created_at', { ascending: false }).limit(20);
  res.json(data || []);
});

// ════════════════════════════════════
// BILLING
// ════════════════════════════════════
app.post('/api/billing/checkout', requireAuth, async (req, res) => {
  const { data: user } = await supabase.from('users').select('email,stripe_customer_id').eq('id', req.user.id).single();
  let customerId = user?.stripe_customer_id;
  if (!customerId) {
    const customer = await stripe.customers.create({ email: user.email, metadata: { userId: req.user.id } });
    customerId = customer.id;
    await supabase.from('users').update({ stripe_customer_id: customerId }).eq('id', req.user.id);
  }
  const session = await stripe.checkout.sessions.create({
    customer: customerId, mode: 'subscription', payment_method_types: ['card'],
    line_items: [{ price: process.env.STRIPE_PRO_PRICE_ID, quantity: 1 }],
    success_url: `${process.env.FRONTEND_URL}?upgraded=true`,
    cancel_url: process.env.FRONTEND_URL,
  });
  res.json({ url: session.url });
});

app.post('/api/billing/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
  let event;
  try { event = stripe.webhooks.constructEvent(req.body, req.headers['stripe-signature'], process.env.STRIPE_WEBHOOK_SECRET); }
  catch { return res.status(400).send('Webhook error'); }
  const customerId = event.data.object.customer;
  if (!customerId) return res.json({ received: true });
  const { data: user } = await supabase.from('users').select('id').eq('stripe_customer_id', customerId).single();
  if (!user) return res.json({ received: true });
  if (['customer.subscription.created','customer.subscription.updated'].includes(event.type)) {
    const plan = event.data.object.status === 'active' ? 'pro' : 'free';
    await supabase.from('users').update({ plan }).eq('id', user.id);
  }
  if (event.type === 'customer.subscription.deleted') {
    await supabase.from('users').update({ plan: 'free' }).eq('id', user.id);
  }
  res.json({ received: true });
});

// ════════════════════════════════════
// HEALTH
// ════════════════════════════════════
app.get('/api/health', (_, res) => res.json({ status: 'ok', version: '4.1.0' }));

app.listen(PORT, () => console.log(`EconoMind API running on :${PORT}`));
